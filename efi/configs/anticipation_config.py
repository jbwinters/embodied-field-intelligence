"""Opt-in temporal control; existing AgentConfig defaults are unchanged."""

from dataclasses import dataclass


@dataclass
class AnticipationConfig:
    horizon: int = 4
    hazard_cost: float = 2.0
    retention: float = 0.9
    prior: float = 0.02
    forecast_mode: str = "learned"  # learned | static | unlearned
    learn_motion: bool = True
    relational_motion: bool = False
    pool_motion: bool = True
    correct_tracks: bool = True
    moving_targets: bool = False
    target_reward: float = 1.0
    target_sweeps: int = 8
    target_terminal_weight: float = 0.5

    def __post_init__(self):
        if not 1 <= self.horizon <= 16:
            raise ValueError("horizon must be between 1 and 16")
        if not 1 <= self.target_sweeps <= 32:
            raise ValueError("target_sweeps must be between 1 and 32")
        if not 0 <= self.target_terminal_weight < 1:
            raise ValueError("target_terminal_weight must be in [0, 1)")
        if self.hazard_cost < 0 or self.target_reward < 0:
            raise ValueError("hazard_cost and target_reward must be nonnegative")
        if not 0 < self.retention <= 1 or self.prior <= 0:
            raise ValueError("invalid motion learning rates")
        if self.forecast_mode not in ("learned", "static", "unlearned"):
            raise ValueError("unknown forecast_mode")
