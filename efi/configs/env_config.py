"""Environment configuration."""

from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Configuration for ForageWorld environment."""
    
    # Grid dimensions
    H: int = 17
    W: int = 17
    
    # Observation window size
    win: int = 5
    
    # World generation
    p_wall: float = 0.12
    n_targets_A: int = 3
    n_targets_B: int = 3
    
    # Episode settings
    max_steps: int = 200
    
    # Rewards and penalties
    step_cost: float = -0.01
    bump_pen: float = -0.02
    reward_A: float = +1.0
    reward_B: float = +0.6
    
    # Random seed
    seed: int = 0