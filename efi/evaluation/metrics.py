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


@dataclass
class ExperimentResults:
    """Results from an experiment run."""
    
    metrics: List[EpisodeMetrics]
    mean_return: float
    std_return: float
    mean_steps: float
    std_steps: float
    config: Dict[str, Any] = field(default_factory=dict)