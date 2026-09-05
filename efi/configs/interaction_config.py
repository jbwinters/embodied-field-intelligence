"""Explicit resource and observation contract for the local contact pilot."""

from dataclasses import dataclass


@dataclass(frozen=True)
class InteractionConfig:
    map_size: int = 31
    horizon: int = 2
    temperature: float = 0.02
    prior: float = 0.01
    retention: float = 0.95
    step_cost: float = -0.01
    collision_cost: float = -2.0
    goal_reward_bound: float = 1.0
    learn: bool = True
    action_conditioned: bool = True
    rule_passes: int = 2

    def __post_init__(self):
        if not isinstance(self.map_size, int) or self.map_size < 9 or self.map_size % 2 != 1:
            raise ValueError("odd internal map of at least 9 cells required")
        if self.horizon not in (1, 2):
            raise ValueError("the contact pilot supports horizons one and two")
        if self.temperature <= 0 or self.prior <= 0:
            raise ValueError("positive temperature and pseudocount required")
        if (
            not 0 < self.retention <= 1
            or not isinstance(self.rule_passes, int)
            or not 0 <= self.rule_passes <= 2
        ):
            raise ValueError("invalid evidence retention or rule transport budget")
        if self.step_cost > 0 or self.collision_cost >= 0:
            raise ValueError("nonpositive step cost and negative collision cost required")
        if self.goal_reward_bound <= 0:
            raise ValueError("positive task reward bound required")
