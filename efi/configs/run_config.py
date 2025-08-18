"""Run/experiment configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RunConfig:
    """Configuration for experiment runs."""
    
    episodes: int = 5
    seeds: int = 1
    render: str = "none"   # none|live
    save_video: Optional[str] = None
    out_dir: str = "runs"
    
    # Ablation flags
    novelty: int = 1
    trail: int = 1
    corner: int = 1
    schema: int = 1
    wall_proximity: int = 1  # Added for A4