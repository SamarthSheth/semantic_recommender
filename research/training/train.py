"""
Contrastive fine-tuning training loop.

Usage:
    python train.py --config baseline
    python train.py --config no_hard_neg
    python train.py --config baseline --no-wandb   # skip W&B logging

This script orchestrates the full training pipeline:
    1. Load training data and create dataloaders
    2. Initialize model from pretrained weights
    3. Train with InfoNCE contrastive loss
    4. Evaluate periodically on validation set
    5. Early stopping when validation loss plateaus
    6. Rebuild FAISS index with fine-tuned embeddings
    7. Run full retrieval evaluation on test set
    8. Save model, metrics, and learning curves

== Training Strategy ==

We use AdamW with linear warmup and cosine decay:

    LR
    ^
    |    /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
    |   /                        \
    |  /                          \
    | /                            \
    +--+-----+---------------------+--> steps
       warmup        cosine decay

- Warmup: prevents catastrophic forgetting in early steps by keeping
  gradients small while the model adjusts to the new loss landscape.
- Cosine decay: gradually reduces LR so the model converges smoothly
  rather than oscillating around the optimum.

This schedule is standard for fine-tuning transformers (used by
HuggingFace, CLIP, etc.).

== What gets saved ==

models/{experiment_name}/
    ├── model/              # The fine-tuned sentence-transformer (loadable)
    ├── training_state.pt   # Optimizer state (for resuming training)
    ├── metrics.json        # All evaluation metrics
    ├── config.json         # Exact config used (reproducibility)
    └── learning_curves.json  # Loss/accuracy per step (for plotting)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "training"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "evaluation"))

from config_loader import load_config
from configs import CONFIGS
from dataset import create_dataloaders
from metrics import RetrievalEvaluator
from model import ContrastiveMovieEncoder, ContrastiveLossWithHardNegatives


def set_seed(seed: int):
    """
    Set all random seeds for reproducibility.

    We set seeds for Python, NumPy, PyTorch CPU, and PyTorch CUDA.
    Also set torch to deterministic mode (slightly slower but reproducible).

    Why this matters: contrastive learning is sensitive to batch composition
    (which items are in each batch determines the negatives). Different
    random seeds → different batches → different results. Fixing seeds
    means re-running an experiment gives the same numbers.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Select the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon GPU — faster than CPU for our model size
        return torch.device("mps")
    return torch.device("cpu")


def build_lr_scheduler(optimizer, config: dict, steps_per_epoch: int):
    """
    Build learning rate scheduler: linear warmup → cosine decay.

    Two-phase schedule:
    Phase 1 (warmup): LR goes from 0 → learning_rate linearly.
        Duration: warmup_fraction * total_steps
        Purpose: Avoid large gradient updates early when the loss
        landscape is unfamiliar to the model.

    Phase 2 (cosine decay): LR goes from learning_rate → 0 following
        a cosine curve. Smoother than step decay, avoids the sudden
        drops that can destabilize training.
    """
    total_steps = steps_per_epoch * config["max_epochs"]
    warmup_steps = int(total_steps * config["warmup_fraction"])
    decay_steps = total_steps - warmup_steps

    warmup = LinearLR(
        optimizer,
        start_factor=0.01,  # Start at 1% of target LR
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=decay_steps,
        eta_min=1e-7,  # Don't go all the way to 0
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )
    return scheduler


def train_one_epoch(
    model: ContrastiveMovieEncoder,
    train_loader,
    optimizer,
    scheduler,
    config: dict,
    global_step: int,
    hn_loss_fn=None,
    wandb_run=None,
) -> tuple[float, float, int, list]:
    """
    Run one epoch of training.

    Returns: (avg_loss, avg_accuracy, updated_global_step, step_logs)
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    step_logs = []

    for batch in train_loader:
        optimizer.zero_grad()

        # Forward pass
        output = model(
            queries=batch["queries"],
            positive_docs=batch["positive_docs"],
            hard_neg_docs=batch["hard_neg_docs"] if config["n_hard_negatives"] > 0 else None,
            hard_neg_counts=batch["hard_neg_counts"] if config["n_hard_negatives"] > 0 else None,
        )

        # Use weighted hard negative loss if configured
        if hn_loss_fn is not None and config["hard_negative_weight"] > 1.0:
            labels = torch.arange(len(batch["queries"]), device=output["similarities"].device)
            loss = hn_loss_fn(
                output["similarities"] * model.temperature,  # undo temperature scaling for re-weighting
                labels,
                len(batch["queries"]),
            )
        else:
            loss = output["loss"]

        # Backward pass
        loss.backward()

        # Gradient clipping — prevents exploding gradients that can
        # destabilize training. Max norm of 1.0 is standard.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Track metrics
        total_loss += loss.item()
        total_acc += output["accuracy"]
        n_batches += 1
        global_step += 1

        step_log = {
            "step": global_step,
            "loss": loss.item(),
            "accuracy": output["accuracy"],
            "temperature": output["temperature"],
            "lr": scheduler.get_last_lr()[0],
        }
        step_logs.append(step_log)

        # Log to W&B
        if wandb_run and global_step % config["log_every_n_steps"] == 0:
            wandb_run.log(step_log)

    avg_loss = total_loss / max(n_batches, 1)
    avg_acc = total_acc / max(n_batches, 1)
    return avg_loss, avg_acc, global_step, step_logs


@torch.no_grad()
def validate(model: ContrastiveMovieEncoder, val_loader, config: dict) -> tuple[float, float]:
    """
    Run validation and return average loss and accuracy.

    We run with torch.no_grad() because:
    1. We don't need gradients for evaluation
    2. Saves ~50% memory (no storing activations for backward pass)
    3. Faster computation
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for batch in val_loader:
        output = model(
            queries=batch["queries"],
            positive_docs=batch["positive_docs"],
            hard_neg_docs=batch["hard_neg_docs"] if config["n_hard_negatives"] > 0 else None,
            hard_neg_counts=batch["hard_neg_counts"] if config["n_hard_negatives"] > 0 else None,
        )

        total_loss += output["loss"].item()
        total_acc += output["accuracy"]
        n_batches += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def rebuild_faiss_index(model: ContrastiveMovieEncoder, movies_df: pd.DataFrame) -> tuple[faiss.Index, list[int]]:
    """
    Rebuild the FAISS index using the fine-tuned encoder.

    After fine-tuning, the embedding space has changed — the old FAISS
    index is stale. We need to re-encode all movies with the updated
    model and build a fresh index.
    """
    print("Rebuilding FAISS index with fine-tuned embeddings...")
    model.eval()

    documents = movies_df["document"].tolist()
    movie_ids = movies_df["tmdbId"].tolist()

    # Encode in batches
    all_embeddings = []
    batch_size = 128
    for i in tqdm(range(0, len(documents), batch_size), desc="Encoding"):
        batch_docs = documents[i : i + batch_size]
        with torch.no_grad():
            embeds = model.encode(batch_docs).cpu().numpy()
        all_embeddings.append(embeds)

    embeddings = np.vstack(all_embeddings).astype(np.float32)

    # Build flat index (exact search — fine for our catalog size)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(f"Built index with {index.ntotal} vectors")

    return index, movie_ids


def main():
    parser = argparse.ArgumentParser(description="Train contrastive movie encoder")
    parser.add_argument(
        "--config", type=str, default="baseline",
        choices=list(CONFIGS.keys()),
        help="Experiment configuration name",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    # Load configs
    exp_config = CONFIGS[args.config].copy()
    if args.no_wandb:
        exp_config["use_wandb"] = False
    project_config = load_config()

    print(f"Experiment: {args.config}")
    print(f"Config: {json.dumps(exp_config, indent=2)}")

    # Reproducibility
    set_seed(exp_config["seed"])
    device = get_device()
    print(f"Device: {device}")

    # ── Setup output directory ───────────────────────────────────────
    output_dir = PROJECT_ROOT / "models" / args.config
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    with open(output_dir / "config.json", "w") as f:
        json.dump(exp_config, f, indent=2)

    # ── Load data ────────────────────────────────────────────────────
    processed_dir = Path(project_config["paths"]["processed_data"])
    data_prep_dir = PROJECT_ROOT / "research" / "data_prep" / "data_prep"

    movies = pd.read_parquet(processed_dir / "movies_final.parquet")
    movie_docs = dict(zip(movies["tmdbId"], movies["document"]))
    print(f"Movie catalog: {len(movies):,} movies")

    train_loader, val_loader = create_dataloaders(
        train_path=data_prep_dir / "train_pairs.parquet",
        val_path=data_prep_dir / "val_pairs.parquet",
        movie_docs=movie_docs,
        batch_size=exp_config["batch_size"],
        n_hard_negatives=exp_config["n_hard_negatives"],
    )
    print(f"Train batches: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")

    # ── Initialize model ─────────────────────────────────────────────
    model = ContrastiveMovieEncoder(
        model_name=exp_config["model_name"],
        temperature=exp_config["initial_temperature"],
        learnable_temperature=exp_config["learnable_temperature"],
    )
    model.to(device)

    # Hard negative loss (only used if weight > 1.0)
    hn_loss_fn = None
    if exp_config["hard_negative_weight"] > 1.0:
        hn_loss_fn = ContrastiveLossWithHardNegatives(
            hard_negative_weight=exp_config["hard_negative_weight"]
        )

    # ── Optimizer ────────────────────────────────────────────────────
    # Separate parameter groups: encoder weights get weight decay,
    # temperature parameter does not (it's a scalar, not a weight matrix)
    optimizer = AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": exp_config["learning_rate"],
                "weight_decay": exp_config["weight_decay"],
            },
            {
                "params": [model.log_temperature],
                "lr": exp_config["learning_rate"] * 10,  # Temperature can learn faster
                "weight_decay": 0.0,
            },
        ]
    )

    scheduler = build_lr_scheduler(optimizer, exp_config, len(train_loader))

    # ── W&B setup ────────────────────────────────────────────────────
    wandb_run = None
    if exp_config["use_wandb"]:
        try:
            import wandb
            wandb_run = wandb.init(
                project="semantic-movie-rec",
                name=args.config,
                config=exp_config,
            )
        except ImportError:
            print("wandb not installed, skipping logging")
            exp_config["use_wandb"] = False

    # ── Training loop ────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    all_step_logs = []
    epoch_logs = []

    print(f"\n{'=' * 70}")
    print("STARTING TRAINING")
    print(f"{'=' * 70}\n")

    global_step = 0
    start_time = time.time()

    for epoch in range(exp_config["max_epochs"]):
        epoch_start = time.time()

        # Train
        train_loss, train_acc, global_step, step_logs = train_one_epoch(
            model, train_loader, optimizer, scheduler, exp_config,
            global_step, hn_loss_fn, wandb_run,
        )
        all_step_logs.extend(step_logs)

        # Validate
        val_loss, val_acc = validate(model, val_loader, exp_config)

        epoch_time = time.time() - epoch_start
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "temperature": model.temperature.item(),
            "lr": scheduler.get_last_lr()[0],
            "epoch_time_sec": epoch_time,
        }
        epoch_logs.append(epoch_log)

        print(
            f"Epoch {epoch+1:>3}/{exp_config['max_epochs']}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
            f"τ={model.temperature.item():.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  "
            f"({epoch_time:.1f}s)"
        )

        if wandb_run:
            wandb_run.log({f"epoch/{k}": v for k, v in epoch_log.items()})

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            model.encoder.save(str(output_dir / "model"))
            torch.save({
                "log_temperature": model.log_temperature.data,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
            }, output_dir / "training_state.pt")
            print(f"  → Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= exp_config["patience"]:
                print(f"\n  Early stopping at epoch {epoch+1} "
                      f"(no improvement for {exp_config['patience']} epochs)")
                break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")

    # ── Save learning curves ─────────────────────────────────────────
    with open(output_dir / "learning_curves.json", "w") as f:
        json.dump({
            "step_logs": all_step_logs,
            "epoch_logs": epoch_logs,
            "total_time_sec": total_time,
        }, f, indent=2)

    # ── Full evaluation on test set ──────────────────────────────────
    print(f"\n{'=' * 70}")
    print("RUNNING TEST SET EVALUATION")
    print(f"{'=' * 70}")

    # Reload best model
    from sentence_transformers import SentenceTransformer
    best_model = SentenceTransformer(str(output_dir / "model"))

    # Rebuild FAISS index with fine-tuned embeddings
    fine_tuned_index, movie_ids = rebuild_faiss_index(model, movies)

    # Load test data
    test_df = pd.read_parquet(data_prep_dir / "test_pairs.parquet")
    print(f"Test set: {len(test_df):,} pairs")

    # Evaluate fine-tuned model
    evaluator = RetrievalEvaluator(best_model, fine_tuned_index, movie_ids, movies)
    results = evaluator.evaluate(test_df, top_k_values=[5, 10, 20])
    evaluator.print_report(results, model_name=f"Fine-tuned ({args.config})")

    # Evaluate baseline for comparison
    print("\nRunning baseline comparison...")
    embeddings_dir = Path(project_config["paths"]["embeddings"])
    baseline_index = faiss.read_index(str(embeddings_dir / "faiss_index.bin"))
    with open(embeddings_dir / "movie_ids.json") as f:
        baseline_movie_ids = json.load(f)

    baseline_model = SentenceTransformer(exp_config["model_name"])
    baseline_evaluator = RetrievalEvaluator(
        baseline_model, baseline_index, baseline_movie_ids, movies
    )
    baseline_results = baseline_evaluator.evaluate(test_df, top_k_values=[5, 10, 20])
    baseline_evaluator.print_report(baseline_results, model_name="Baseline (frozen)")

    # Statistical comparison
    from metrics import compare_models
    compare_models(baseline_results, results, "Baseline", f"Fine-tuned ({args.config})")

    # Save all metrics
    with open(output_dir / "metrics.json", "w") as f:
        # Remove per_query (too large) but save summary
        results_summary = {k: v for k, v in results.items() if k != "per_query"}
        baseline_summary = {k: v for k, v in baseline_results.items() if k != "per_query"}
        json.dump({
            "fine_tuned": results_summary,
            "baseline": baseline_summary,
            "config": exp_config,
        }, f, indent=2)

    if wandb_run:
        wandb_run.finish()

    print(f"\nAll outputs saved to {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
