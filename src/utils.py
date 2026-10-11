"""Utility functions for seeding, configuration, and language detection."""

import os
import random
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

# Unicode codepoint ranges for Indic scripts
DEVANAGARI_RANGE = (0x0900, 0x097F)
BENGALI_RANGE = (0x0980, 0x09FF)


def load_config(config_path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load configuration YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing parsed configuration.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42) -> None:
    """Set global random seed across standard libraries for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def guess_language(text: str) -> str:
    """Guess language based on Unicode script character counts.

    Identifies Hindi (Devanagari script), Bengali (Bengali script), or
    categorizes as 'other' for Romanized / English / non-Indic scripts.

    Args:
        text: Input string.

    Returns:
        'hindi', 'bengali', or 'other'.
    """
    if not isinstance(text, str) or not text.strip():
        return "other"

    hindi_count = sum(
        1 for char in text if DEVANAGARI_RANGE[0] <= ord(char) <= DEVANAGARI_RANGE[1]
    )
    bengali_count = sum(
        1 for char in text if BENGALI_RANGE[0] <= ord(char) <= BENGALI_RANGE[1]
    )

    if hindi_count > 0 and hindi_count >= bengali_count:
        return "hindi"
    if bengali_count > 0 and bengali_count > hindi_count:
        return "bengali"
    return "other"
