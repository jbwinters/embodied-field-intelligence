"""Agent configuration."""

from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Configuration for ChemotaxisAgentCA."""
    
    # Scent field parameters
    seed_strength: float = 1.0  # Stronger scent signal
    scent_diff: float = 0.25    # Much more diffusion for longer range
    scent_decay: float = 0.005   # Very slow decay
    scent_steps: int = 4         # More diffusion steps
    
    # Visit trail parameters
    v_inj: float = 1.0
    v_decay: float = 0.02  # Slower fade
    v_diff: float = 0.08   # Tighter trail
    k_repulse: float = 0.30
    
    # Exploration parameters
    wander: float = 0.0    # We moved noise to action sampler
    stay_thresh: float = 0.02
    
    # Anti-stuck mechanism
    anti_stuck_after: int = 2
    anti_stuck_temp: float = 0.8  # Higher for decisive escapes
    
    # Internal processing
    internal_think: int = 0  # extra diffusion ticks per step
    
    # Random seed
    seed: int = 0


@dataclass
class Ablations:
    """Feature ablation flags."""
    
    trail: int = 1
    novelty: int = 1
    corner: int = 1
    schema: int = 1