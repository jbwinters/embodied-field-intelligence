"""Agent configuration."""

from dataclasses import dataclass, field

from .belief_config import BeliefConfig


@dataclass
class AgentConfig:
    """Configuration for ChemotaxisAgentCA."""

    # Belief fields (log-odds Bayes filter replacing raw scent; FieldController only)
    use_belief_fields: bool = True
    belief: BeliefConfig = field(default_factory=BeliefConfig)

    # LMDP control: value as a local fixed point of z-iteration (FieldController only)
    control_mode: str = "lmdp"  # "lmdp" | "legacy" (potential composition)
    # lam sets BOTH planning risk and action softmax temperature. It must be
    # small relative to reward scale: the LMDP charges a KL control cost of
    # lam*log(4) per step, so value reach is ~ r / (q_step + lam*log 4).
    # lam=0.02 -> reach ~ 26 cells per unit reward.
    lam_base: float = 0.02
    z_sweeps: int = 3           # value-iteration sweeps per env tick (kappa)
    init_sweeps: int = 0        # extra sweeps on the first tick of an episode
                                # (0 = auto: H+W, "orienting before moving")
    # State costs, units of reward-per-step. These are PATH costs the
    # planner routes around -- deliberately milder than the legacy
    # potential-subtraction weights (a 1.8/step trail cost would price a
    # corridor above the total reward and trap the agent behind its own path).
    q_step: float = 0.01        # effort per step (matches env step cost scale)
    q_trail: float = 0.08       # transit cost of recently visited cells
    q_corner: float = 0.02
    q_wall_prox: float = 0.02
    q_pain: float = 0.3
    q_membrane: float = 0.3

    # Affect -> lambda (one dial replaces temperature + semiring flip):
    # pain lowers lambda toward the max-plus/worst-case limit; arousal
    # raises it mildly. Same lambda drives value sweeps AND action softmax.
    k_pain_lambda: float = 0.9
    k_arousal_lambda: float = 0.3
    lam_min: float = 0.005
    lam_max: float = 0.1

    # Exact barrier: membrane cells at/above this level get q = +inf
    # (V = -VBIG), so the softmax selects them with probability exactly 0.
    barrier_threshold: float = 0.75

    # Epistemic term in the reward injection (lmdp mode):
    #   "infogain": beta * meanpool_win(belief entropy + map uncertainty),
    #               affect-modulated (curiosity raises beta, fear lowers it)
    #   "frontier": legacy diffused-unseen attractor (needs no beliefs)
    #   "none":     pure exploitation (the honest ablation)
    epistemic_mode: str = "infogain"
    beta_epist: float = 0.3
    k_curiosity: float = 0.5
    k_fear: float = 0.8
    w_map_uncertainty: float = 1.0

    # Field pyramid: 1 = single scale (off), 2 = use a half-resolution
    # level to accelerate the COLD-START convergence at episode start
    # (coarse sweeps are cheap horizon). Measured: as an every-tick bound
    # it injects optimism bias through coarsened walls and mildly hurts;
    # as a one-shot initializer it provably speeds convergence. With
    # warm-started tracking, steady-state behavior at <=50x50 is already
    # saturated, so this stays off by default.
    pyramid_levels: int = 1

    # Schema: "predictive" (count-based local world-rule learner; its error
    # is the surprise signal, its static-confidence gates belief blur),
    # "oja" (legacy Oja/BCM prototypes via the runner), or "off".
    schema_mode: str = "predictive"

    # Egocentric controller: internal map edge (must exceed any world the
    # agent will meet: 2*64+1 covers 64 steps of travel in any direction
    # from the starting pose). A production agent would scroll the map.
    map_size: int = 129
    # Loop closure: correlate observed walls against the internal map at
    # +/-1 pose offsets; snap when a shift wins by >= 2 matching cells.
    pose_correction: bool = True

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