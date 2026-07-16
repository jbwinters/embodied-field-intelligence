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
    n_targets_A: int = 2
    n_targets_B: int = 4  # More B targets for faster learning signal
    
    # Episode settings
    max_steps: int = 200
    
    # Rewards and penalties
    step_cost: float = -0.01
    bump_pen: float = -0.02
    reward_A: float = +1.0
    reward_B: float = -0.8  # Now aversive!

    # Odometry slip probability: successful moves land on a random other
    # passable neighbor with this probability, while info["moved"] stays
    # True (egocentric dead reckoning must detect the drift).
    p_slip: float = 0.0

    # --- Non-stationarity (all off by default) ---
    # regrow: a picked-up target respawns at a random free cell after a
    # Geometric(p_regrow) delay. With regrow on, the episode only ends at
    # the step cap.
    p_regrow: float = 0.0
    # drift: every T_shift steps (0 = off), each remaining target teleports
    # w.p. p_move to a random free cell within Chebyshev radius r_drift.
    T_shift: int = 0
    p_move: float = 0.5
    r_drift: int = 4
    # swap: at step T_swap (0 = off), reward_A and reward_B swap values --
    # the mid-episode revaluation test.
    T_swap: int = 0

    # Random seed
    seed: int = 0