"""General utilities."""

import random
import time
from pathlib import Path

import numpy as np


def set_global_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))


def ts():
    """Get current timestamp string."""
    return time.strftime("%Y%m%d-%H%M%S")


def ensure_dir(p: str | Path) -> Path:
    """Ensure directory exists, creating if necessary."""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p