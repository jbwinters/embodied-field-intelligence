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

    def __post_init__(self):
        if not 1 <= self.horizon <= 16:
            raise ValueError("horizon must be between 1 and 16")
        if self.hazard_cost < 0:
            raise ValueError("hazard_cost must be nonnegative")
        if not 0 < self.retention <= 1 or self.prior <= 0:
            raise ValueError("invalid motion learning rates")
        if self.forecast_mode not in ("learned", "static", "unlearned"):
            raise ValueError("unknown forecast_mode")
