"""
Shared configuration loader.

Why a separate module: every script needs config values, and we want
a single source of truth. This also validates that required keys exist
at import time rather than failing deep inside a pipeline step.
"""

import yaml
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """Load and validate the project config."""
    if config_path is None:
        # Walk up from this file to find the project root
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate required sections exist
    required_sections = ["data", "model", "evaluation", "paths"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    for key, rel_path in config["paths"].items():
        config["paths"][key] = str(project_root / rel_path)
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)

    return config
