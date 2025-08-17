"""Schema field configuration."""

from dataclasses import dataclass


@dataclass
class SchemaConfig:
    """Configuration for SchemaField learning."""
    
    # Enable/disable schema field
    enabled: int = 1
    
    # Tile parameters
    tile: int = 5          # tile size (patch for prototypes)
    K: int = 4             # prototypes per tile
    
    # Learning parameters
    eta: float = 0.03      # learning rate (Oja/BCM)
    slowness: float = 0.02 # penalty on |y - y_prev|
    bcm_tau: float = 0.01  # running avg rate for theta (postsyn. threshold)
    comp_k: int = 1        # number of winners per tile (soft WTA)
    
    # Diffusion parameters
    diff: float = 0.08     # diffusion for schema activation fields
    decay: float = 0.005
    steps: int = 1
    
    # Control influence
    alpha_schema: float = 0.35  # strength into P_eff (positive bias)
    
    # Random seed
    seed: int = 0