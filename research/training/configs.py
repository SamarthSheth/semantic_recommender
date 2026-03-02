"""
Training experiment configurations.

Each config represents a different experiment. The baseline config
should be run first, then ablations change one variable at a time
to measure the effect of each design choice.

Ablation strategy (run in this order):
1. baseline    → establishes the starting point with reasonable defaults
2. no_hard_neg → removes hard negatives. Measures their contribution.
3. high_lr     → 5x higher learning rate. Tests if we're undertraining.
4. low_lr      → 5x lower learning rate. Tests if we're overtraining.
5. fixed_temp  → non-learnable temperature. Tests if learning τ helps.
6. large_model → uses all-mpnet-base-v2 (768-dim). Tests model capacity.
7. weighted_hn → applies extra weight to hard negatives. Tests if
                 harder training signal helps.

Each ablation isolates ONE variable. This is important — changing
multiple things at once makes it impossible to attribute improvements.
"""

CONFIGS = {
    "baseline": {
        # Model
        "model_name": "all-MiniLM-L6-v2",
        "learnable_temperature": True,
        "initial_temperature": 0.07,

        # Training
        "batch_size": 64,
        "learning_rate": 2e-5,          # Low LR because we're fine-tuning,
                                         # not training from scratch. We want
                                         # to nudge the pretrained weights,
                                         # not overwrite them.
        "weight_decay": 0.01,            # L2 regularization. Prevents any
                                         # single weight from growing too large.
                                         # Standard value for transformer fine-tuning.
        "warmup_fraction": 0.1,          # Linearly increase LR from 0 to target
                                         # over the first 10% of training. Prevents
                                         # large, destructive gradient updates before
                                         # the model has "warmed up" to the new task.
        "max_epochs": 20,
        "patience": 5,                   # Early stopping: stop if val loss hasn't
                                         # improved in 5 epochs. With only ~13K
                                         # training pairs, overfitting is a real risk.

        # Data
        "n_hard_negatives": 4,
        "hard_negative_weight": 1.0,     # 1.0 = no extra weighting (standard InfoNCE)

        # Logging
        "eval_every_n_steps": 100,       # Run validation metrics every 100 training steps
        "log_every_n_steps": 10,         # Log loss/accuracy every 10 steps
        "use_wandb": True,

        # Reproducibility
        "seed": 42,
    },

    "no_hard_neg": {
        # Same as baseline but without hard negatives
        # If this performs similarly, hard negatives aren't helping
        "model_name": "all-MiniLM-L6-v2",
        "learnable_temperature": True,
        "initial_temperature": 0.07,
        "batch_size": 64,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_fraction": 0.1,
        "max_epochs": 20,
        "patience": 5,
        "n_hard_negatives": 0,           # ← THE CHANGE
        "hard_negative_weight": 1.0,
        "eval_every_n_steps": 100,
        "log_every_n_steps": 10,
        "use_wandb": True,
        "seed": 42,
    },

    "high_lr": {
        "model_name": "all-MiniLM-L6-v2",
        "learnable_temperature": True,
        "initial_temperature": 0.07,
        "batch_size": 64,
        "learning_rate": 1e-4,           # ← THE CHANGE (5x higher)
        "weight_decay": 0.01,
        "warmup_fraction": 0.1,
        "max_epochs": 20,
        "patience": 5,
        "n_hard_negatives": 4,
        "hard_negative_weight": 1.0,
        "eval_every_n_steps": 100,
        "log_every_n_steps": 10,
        "use_wandb": True,
        "seed": 42,
    },

    "low_lr": {
        "model_name": "all-MiniLM-L6-v2",
        "learnable_temperature": True,
        "initial_temperature": 0.07,
        "batch_size": 64,
        "learning_rate": 4e-6,           # ← THE CHANGE (5x lower)
        "weight_decay": 0.01,
        "warmup_fraction": 0.1,
        "max_epochs": 20,
        "patience": 5,
        "n_hard_negatives": 4,
        "hard_negative_weight": 1.0,
        "eval_every_n_steps": 100,
        "log_every_n_steps": 10,
        "use_wandb": True,
        "seed": 42,
    },

    "fixed_temp": {
        "model_name": "all-MiniLM-L6-v2",
        "learnable_temperature": False,  # ← THE CHANGE
        "initial_temperature": 0.07,
        "batch_size": 64,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_fraction": 0.1,
        "max_epochs": 20,
        "patience": 5,
        "n_hard_negatives": 4,
        "hard_negative_weight": 1.0,
        "eval_every_n_steps": 100,
        "log_every_n_steps": 10,
        "use_wandb": True,
        "seed": 42,
    },

    "large_model": {
        "model_name": "all-mpnet-base-v2",  # ← THE CHANGE (768-dim, more capacity)
        "learnable_temperature": True,
        "initial_temperature": 0.07,
        "batch_size": 32,                # Smaller batch — larger model uses more memory
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_fraction": 0.1,
        "max_epochs": 20,
        "patience": 5,
        "n_hard_negatives": 4,
        "hard_negative_weight": 1.0,
        "eval_every_n_steps": 100,
        "log_every_n_steps": 10,
        "use_wandb": True,
        "seed": 42,
    },

    "weighted_hn": {
        "model_name": "all-MiniLM-L6-v2",
        "learnable_temperature": True,
        "initial_temperature": 0.07,
        "batch_size": 64,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_fraction": 0.1,
        "max_epochs": 20,
        "patience": 5,
        "n_hard_negatives": 4,
        "hard_negative_weight": 2.0,     # ← THE CHANGE (hard negatives count 2x)
        "eval_every_n_steps": 100,
        "log_every_n_steps": 10,
        "use_wandb": True,
        "seed": 42,
    },
}
