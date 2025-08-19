"""Metrics and results tracking."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    
    total_return: float
    steps: int
    targets_collected: Dict[str, int] = field(default_factory=dict)
    efficiency: float = 0.0
    seed: int = 0
    episode: int = 0
    mean_cosine: Optional[float] = None  # Alignment between gradient and motion
    valence_snapshot: Dict[str, float] = field(default_factory=dict)  # Current valence weights
    
    # Safety metrics
    bumps_per_100: float = 0.0  # Bumps per 100 steps
    mean_pain: float = 0.0  # Average pain level
    max_pain: float = 0.0  # Maximum pain level
    mean_wall_distance: float = 0.0  # Average distance to nearest wall
    affect_history: List[Dict[str, float]] = field(default_factory=list)  # Full affect state history


@dataclass
class ExperimentResults:
    """Results from an experiment run."""
    
    metrics: List[EpisodeMetrics]
    mean_return: float
    std_return: float
    mean_steps: float
    std_steps: float
    config: Dict[str, Any] = field(default_factory=dict)