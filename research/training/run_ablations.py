"""
Run all ablation experiments sequentially and produce a comparison table.

Usage:
    python run_ablations.py                    # Run all experiments
    python run_ablations.py --only baseline no_hard_neg   # Run subset
    python run_ablations.py --skip-training    # Just compare existing results

This is the script you run once and walk away. It:
1. Trains each experiment configuration
2. Evaluates each on the same test set
3. Produces a summary table comparing all experiments
4. Saves the table as a CSV for inclusion in your writeup

The summary table is the centerpiece of your research section.
A quant firm wants to see: "I changed X, performance changed by Y,
with statistical significance Z." This script produces exactly that.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent

sys.path.insert(0, str(Path(__file__).parent))
from configs import CONFIGS


def run_experiment(config_name: str, no_wandb: bool = False):
    """Run a single training experiment as a subprocess."""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "train.py"),
        "--config", config_name,
    ]
    if no_wandb:
        cmd.append("--no-wandb")

    print(f"\n{'#' * 70}")
    print(f"# RUNNING EXPERIMENT: {config_name}")
    print(f"{'#' * 70}\n")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"WARNING: Experiment {config_name} failed with code {result.returncode}")
        return False
    return True


def collect_results() -> pd.DataFrame:
    """Load metrics.json from each completed experiment into a summary table."""
    rows = []
    models_dir = PROJECT_ROOT / "models"

    for config_name in CONFIGS:
        metrics_path = models_dir / config_name / "metrics.json"
        if not metrics_path.exists():
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        ft = metrics["fine_tuned"]
        bl = metrics["baseline"]

        row = {
            "experiment": config_name,
            # Fine-tuned metrics
            "ft_mrr": ft["overall"]["mrr"]["mean"],
            "ft_hit@5": ft["by_k"]["5"]["hit_rate"]["mean"],
            "ft_hit@10": ft["by_k"]["10"]["hit_rate"]["mean"],
            "ft_ndcg@10": ft["by_k"]["10"]["ndcg"]["mean"],
            "ft_ndcg@20": ft["by_k"]["20"]["ndcg"]["mean"],
            # Baseline metrics (should be same across experiments)
            "bl_mrr": bl["overall"]["mrr"]["mean"],
            "bl_ndcg@10": bl["by_k"]["10"]["ndcg"]["mean"],
            # Deltas
            "Δ_mrr": ft["overall"]["mrr"]["mean"] - bl["overall"]["mrr"]["mean"],
            "Δ_ndcg@10": ft["by_k"]["10"]["ndcg"]["mean"] - bl["by_k"]["10"]["ndcg"]["mean"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("ft_ndcg@10", ascending=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", help="Only run these experiments")
    parser.add_argument("--skip-training", action="store_true", help="Just collect results")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    experiments = args.only if args.only else list(CONFIGS.keys())

    if not args.skip_training:
        # Always run baseline first
        if "baseline" in experiments:
            experiments.remove("baseline")
            experiments.insert(0, "baseline")

        for config_name in experiments:
            success = run_experiment(config_name, no_wandb=args.no_wandb)
            if not success and config_name == "baseline":
                print("Baseline failed — aborting. Fix errors before running ablations.")
                sys.exit(1)

    # Collect and display results
    print(f"\n{'=' * 70}")
    print("ABLATION SUMMARY")
    print(f"{'=' * 70}\n")

    df = collect_results()
    if len(df) == 0:
        print("No results found. Run experiments first.")
        return

    # Format for display
    display_cols = [
        "experiment", "ft_mrr", "ft_hit@10", "ft_ndcg@10", "Δ_mrr", "Δ_ndcg@10"
    ]
    print(df[display_cols].to_string(index=False, float_format="%.4f"))

    # Save full table
    output_path = PROJECT_ROOT / "research" / "evaluation" / "ablation_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
