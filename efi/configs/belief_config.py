"""Belief field configuration."""

from dataclasses import dataclass


@dataclass
class BeliefConfig:
    """
    Configuration for log-odds Bayesian belief fields (efi/core/belief.py).

    Defaults assume noiseless observations (ForageWorld): evidence increments
    (+/-12) exceed the clamp (+/-8), so a single observation saturates belief
    in either direction regardless of its prior state -- hard evidence.
    Soften l_pos/l_neg below the clamp for noisy-observation environments.
    """

    l_pos: float = 12.0      # evidence added when a target is observed
    l_neg: float = -12.0     # evidence added when a cell is observed empty
    l_prior: float = -4.0    # resting log-odds (targets are rare: p ~ 0.018)
    rho_prior: float = 0.002  # per-tick relaxation toward the prior
    belief_diff: float = 0.14  # probability-space diffusion rate per tick
    belief_decay: float = 0.0  # probability decay per tick (0 = conserve mass)
    l_min: float = -8.0      # saturation clamp (must stay disconfirmable)
    l_max: float = 8.0
