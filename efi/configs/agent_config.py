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
    
    # Valence learning (A desirable, B undesirable will be learned online)
    valA_init: float = 0.10
    valB_init: float = 0.10     # start neutral-ish so behavior emerges
    valence_lr: float = 0.25    # how fast to adapt (tune 0.1–0.4)
    valence_clip: float = 1.5   # keep weights in a sane range
    
    # Field weights (to avoid magic numbers)
    w_novel: float = 0.7        # Novelty attraction weight
    w_trail: float = 0.6        # Trail repulsion weight  
    w_corner: float = 0.5       # Corner hazard repulsion weight
    
    # Random seed
    seed: int = 0
    
    # Affect system parameters
    affect_enabled: bool = True
    affect_rho_v: float = 0.02     # Valence EWMA decay rate
    affect_rho_a: float = 0.05     # Arousal EWMA decay rate  
    affect_rho_c: float = 0.05     # Control EWMA decay rate
    affect_rho_p: float = 0.1      # Pain EWMA decay rate
    
    # Pain parameters
    w_pain: float = 0.7             # Pain field weight as repulsor
    pain_to_temp_gain: float = 0.6  # Pain to temperature conversion gain
    pain_semiring_threshold: float = 0.6  # Threshold to switch to max-plus mode
    
    # Nociception weights
    pain_bump_weight: float = 0.5   # Weight for bump contribution to pain
    pain_reward_weight: float = 0.3 # Weight for negative reward contribution
    pain_prox_weight: float = 0.1   # Weight for wall proximity contribution
    pain_stuck_weight: float = 0.1  # Weight for stuck contribution
    
    # Membrane parameters
    membrane_enabled: bool = True
    w_membrane: float = 0.6         # Membrane field weight
    membrane_r_min: float = 1.0     # Minimum membrane radius
    membrane_r_gain_arousal: float = 1.0  # Arousal contribution to radius
    membrane_r_gain_pain: float = 1.5     # Pain contribution to radius
    
    # Brain membrane (learning gate) parameters
    brain_membrane_enabled: bool = True
    brain_membrane_suppress: float = 0.5  # Pain suppression factor for learning
    brain_membrane_min_rate: float = 0.1  # Minimum learning rate


@dataclass
class Ablations:
    """Feature ablation flags."""
    
    trail: int = 1
    novelty: int = 1
    corner: int = 1
    schema: int = 1
    wall_proximity: int = 1  # Added for A4