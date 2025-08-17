"""Agent configuration."""

from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Configuration for ChemotaxisAgentCA."""
    
    # Scent field parameters
    seed_strength: float = 0.6
    scent_diff: float = 0.16  # Increased for better gradients
    scent_decay: float = 0.01
    scent_steps: int = 3  # More steps for smoother gradients
    
    # Visit trail parameters
    v_inj: float = 1.0
    v_decay: float = 0.03
    v_diff: float = 0.10
    k_repulse: float = 0.30
    
    # Exploration parameters
    wander: float = 0.12  # Increased to prevent corner trapping
    stay_thresh: float = 0.02
    
    # Anti-stuck mechanism
    anti_stuck_after: int = 2  # Trigger sooner
    anti_stuck_temp: float = 0.6
    
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