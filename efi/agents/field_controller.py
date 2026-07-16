"""General field-based controller using channel-agnostic potential composition."""

from typing import Dict, Optional, Tuple
from collections import deque
import numpy as np

from ..core import diffuse_masked, update_visit_trail, update_novelty
from ..core.belief import sigmoid, logodds_correct, logodds_predict
from ..core.desirability import VBIG, value_sweeps
from ..core.potential import compose_potential
from ..configs.belief_config import BeliefConfig
from .adapters import ControllerAdapter


class FieldController:
    """
    General field-based controller that composes arbitrary attractor/repulsor channels.
    
    This controller:
    - Maintains arbitrary named fields (attractors and repulsors)
    - Learns valences (weights) per channel through experience
    - Composes a single potential field from all influences
    - Provides action selection based on potential gradient
    
    The controller is environment-agnostic through the adapter interface.
    """
    
    def __init__(self, env, adapter: ControllerAdapter, cfg, ablate, seed=0):
        """
        Initialize field controller.
        
        Args:
            env: Environment instance
            adapter: Adapter for environment-specific mappings
            cfg: Configuration object
            ablate: Ablation flags
            seed: Random seed
        """
        self.env = env
        self.adapter = adapter
        self.cfg = cfg
        self.ablate = ablate
        self.rng = np.random.RandomState(seed)
        
        self.H, self.W = env.H, env.W
        self.win = getattr(env, "win", 1)
        
        # Initialize fast fields
        self.fields = {
            "A": np.zeros((self.H, self.W), dtype=np.float32),
            "B": np.zeros((self.H, self.W), dtype=np.float32),
            "Novel": np.zeros((self.H, self.W), dtype=np.float32),
            "Trail": np.zeros((self.H, self.W), dtype=np.float32),
            "Frontier": np.zeros((self.H, self.W), dtype=np.float32),
        }
        
        # Known walls for diffusion blocking (discovered over time)
        self.known_walls = np.zeros((self.H, self.W), dtype=bool)
        self.seen = np.zeros((self.H, self.W), dtype=bool)

        # Log-odds belief fields (Bayes filter over target locations).
        # When enabled, self.fields["A"/"B"] hold sigmoid(L) probability maps.
        self.use_beliefs = bool(getattr(cfg, "use_belief_fields", False))
        self.belief_cfg = getattr(cfg, "belief", None) or BeliefConfig()
        self.L = {
            "A": np.full((self.H, self.W), self.belief_cfg.l_prior, dtype=np.float32),
            "B": np.full((self.H, self.W), self.belief_cfg.l_prior, dtype=np.float32),
        }
        if self.use_beliefs:
            self.fields["A"] = sigmoid(self.L["A"])
            self.fields["B"] = sigmoid(self.L["B"])

        # LMDP value state (warm-started across ticks; see compose_value)
        self.control_mode = str(getattr(cfg, "control_mode", "legacy"))
        self.lam = float(getattr(cfg, "lam_base", 0.5))
        self.z_sweeps = int(getattr(cfg, "z_sweeps", 3))
        self.V = np.zeros((self.H, self.W), dtype=np.float32)
        self.last_residuals = []
        self._fresh_value = True

        # Predictive schema: learns the world's local rule online; its
        # error is the surprise signal and its static-confidence gates
        # belief blur ("memory sharpens once the world is learned static").
        self.pschema = None
        if str(getattr(cfg, "schema_mode", "predictive")) == "predictive":
            from .predictive_schema import PredictiveSchema
            self.pschema = PredictiveSchema(win=self.win)

        # Learnable valence table
        self.valence = {
            "A": float(cfg.valA_init),
            "B": float(cfg.valB_init),
            "Novel": float(cfg.w_novel),  # Can optionally learn slowly
        }
        
        # Backwards compatibility attributes
        self.valA = self.valence["A"]
        self.valB = self.valence["B"]
        
        # State tracking
        self.prev_P_here = 0.0
        self._prev_patch = None
        self._pos_hist = deque(maxlen=3)
        self._pos_hist.append((self.env.y, self.env.x))
        self.last_action = None
        self.stuck_count = 0
        
    def reset(self):
        """Reset controller state for new episode."""
        # Clear all fields
        for k in self.fields:
            self.fields[k][:] = 0.0

        self.known_walls[:] = False
        self.seen[:] = False

        # Reset beliefs to the prior
        for k in self.L:
            self.L[k][:] = self.belief_cfg.l_prior
        if self.use_beliefs:
            self.fields["A"] = sigmoid(self.L["A"])
            self.fields["B"] = sigmoid(self.L["B"])

        # Reset LMDP value state (cold start for the new episode)
        self.V[:] = 0.0
        self.last_residuals = []
        self._fresh_value = True
        
        # Mark initial visible area as seen
        y, x = self.env.y, self.env.x
        half = self.win // 2
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                gy, gx = y + dy, x + dx
                if 0 <= gy < self.H and 0 <= gx < self.W:
                    self.seen[gy, gx] = True
        
        # Reset state tracking
        self.prev_P_here = 0.0
        self._prev_patch = None
        self._pos_hist.clear()
        self._pos_hist.append((self.env.y, self.env.x))
        self.last_action = None
        self.stuck_count = 0
        
    def learn_valence(self, channel: str, reward: float):
        """
        Update valence weight for a channel based on reward.
        
        Args:
            channel: Name of the channel to update
            reward: Reward signal for learning
        """
        lr = float(self.cfg.valence_lr)
        clip = float(getattr(self.cfg, "valence_clip", 1.5))
        current = self.valence.get(channel, 0.0)
        self.valence[channel] = float(np.clip(current + lr * reward, -clip, clip))
        
        # Update backwards compatibility attributes
        if channel == "A":
            self.valA = self.valence["A"]
        elif channel == "B":
            self.valB = self.valence["B"]

    def notify_pickup(self, channel: str):
        """
        Register that a target was just picked up at the agent's cell.

        The target is gone, so belief there collapses to "certainly absent".
        """
        if not self.use_beliefs or channel not in self.L:
            return
        y, x = self.env.y, self.env.x
        self.L[channel][y, x] = self.belief_cfg.l_min
        self.fields[channel][y, x] = sigmoid(np.float32(self.belief_cfg.l_min))

    def step_fields(self, obs_vec: np.ndarray) -> np.ndarray:
        """
        Update all fields based on observation.
        
        Args:
            obs_vec: Flattened observation vector
            
        Returns:
            Walls mask for diffusion operations
        """
        y, x = self.env.y, self.env.x
        
        # 1) Get seeds from observation via adapter
        seeds = self.adapter.seed_from_obs(obs_vec, y, x)
        
        # Update known walls and mark area as seen
        # Infer channel count from observation length (supports different adapters)
        win = self.win
        ch = int(len(obs_vec) // (win * win))
        patch = obs_vec[:ch * win * win].reshape(ch, win, win)
        half = self.win // 2
        
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                gy, gx = y + dy, x + dx
                if 0 <= gy < self.H and 0 <= gx < self.W:
                    self.seen[gy, gx] = True
        
        for (gy, gx) in seeds.get("walls", []):
            self.known_walls[gy, gx] = True

        walls_mask = self.known_walls.copy()

        if self.use_beliefs:
            # 1b/2) Bayes filter over target locations.
            # Correction: positive evidence at observed targets, NEGATIVE
            # evidence at cells observed empty (scent could never disconfirm).
            bc = self.belief_cfg
            walls_local = patch[0] > 0.5
            A_local = patch[1] > 0.5
            B_local = patch[2] > 0.5
            pos_cells = {"A": [], "B": []}
            neg_cells = {"A": [], "B": []}
            for dy in range(-half, half + 1):
                for dx in range(-half, half + 1):
                    gy, gx = y + dy, x + dx
                    if not (0 <= gy < self.H and 0 <= gx < self.W):
                        continue
                    py, px = dy + half, dx + half
                    if walls_local[py, px]:
                        continue  # walls cannot hold targets; no belief there
                    (pos_cells["A"] if A_local[py, px] else neg_cells["A"]).append((gy, gx))
                    (pos_cells["B"] if B_local[py, px] else neg_cells["B"]).append((gy, gx))
            # Belief blur gate: to the extent the agent has LEARNED the
            # world is static, its memory should not diffuse or relax.
            static_conf = self.pschema.static_confidence if self.pschema else 0.0
            gate = 1.0 - float(static_conf)
            for channel in ("A", "B"):
                Lc = logodds_correct(
                    self.L[channel], pos_cells[channel], neg_cells[channel],
                    l_pos=bc.l_pos, l_neg=bc.l_neg, l_min=bc.l_min, l_max=bc.l_max
                )
                Lc = logodds_predict(
                    Lc, walls_mask,
                    diff=bc.belief_diff * gate, decay=bc.belief_decay,
                    l_prior=bc.l_prior, rho_prior=bc.rho_prior * gate
                )
                self.L[channel] = Lc
                # Downstream composition consumes probability maps; learned
                # valences convert probability into subjective value.
                self.fields[channel] = sigmoid(Lc)
        else:
            # Legacy scent path: max-inject seeding + diffusion
            for (gy, gx) in seeds.get("A", []):
                self.fields["A"][gy, gx] = max(self.fields["A"][gy, gx], self.cfg.seed_strength)
            for (gy, gx) in seeds.get("B", []):
                self.fields["B"][gy, gx] = max(self.fields["B"][gy, gx], self.cfg.seed_strength)

            # 2) Diffuse attractor channels
            for channel in ("A", "B"):
                self.fields[channel] = diffuse_masked(
                    self.fields[channel], walls_mask,
                    diff=self.cfg.scent_diff,
                    decay=self.cfg.scent_decay,
                    steps=self.cfg.scent_steps
                )
            
        # 3) Trail (repulsor) with ping-pong detection
        self._pos_hist.append((y, x))
        pingpong = (len(self._pos_hist) == 3 and
                   self._pos_hist[0] == self._pos_hist[2] and
                   self._pos_hist[0] != self._pos_hist[1])
        
        if self.ablate.trail:
            v_inj = self.cfg.v_inj * (2.5 if pingpong else 1.0)
            self.fields["Trail"] = update_visit_trail(
                self.fields["Trail"], y, x, walls_mask,
                v_decay=self.cfg.v_decay,
                v_diff=self.cfg.v_diff,
                v_inj=v_inj
            )
        else:
            self.fields["Trail"][:] = 0.0
            
        # 4) Novelty (attractor) based on prediction error
        P_local = float(self.fields["A"][y, x] + self.fields["B"][y, x])
        d_scent = abs(P_local - self.prev_P_here)
        self.prev_P_here = P_local
        
        # Observation change
        d_obs = 0.0
        if self._prev_patch is not None:
            d_obs = float(np.mean(np.abs(patch[:3] - self._prev_patch[:3])))

        # Predictive schema: learn the transition (prev window, last action)
        # -> this window; its error REPLACES the hand-crafted pred_err.
        if (self.pschema is not None and self._prev_patch is not None
                and self.last_action is not None):
            moved = len(self._pos_hist) >= 2 and self._pos_hist[-2] != (y, x)
            pred_err = self.pschema.observe_transition(
                self._prev_patch, int(self.last_action), bool(moved), patch)
        else:
            pred_err = 0.5 * d_scent + 0.5 * d_obs
        self._prev_patch = patch.copy()
        
        if self.ablate.novelty:
            self.fields["Novel"] = update_novelty(
                self.fields["Novel"], pred_err, y, x, walls_mask,
                n_decay=0.018, n_diff=0.06, gain=6.0
            )
        else:
            self.fields["Novel"][:] = 0.0
            
        # 5) Frontier field (unseen areas)
        U = (~self.seen).astype(np.float32)
        U = U * (~walls_mask).astype(np.float32)  # Mask out walls
        U = diffuse_masked(U, walls_mask, diff=0.15, decay=0.01, steps=3)
        self.fields["Frontier"] = U
        
        return walls_mask
        
    def compose_P(self, walls_mask: np.ndarray,
                  corner_field: Optional[np.ndarray] = None,
                  wall_prox_field: Optional[np.ndarray] = None,
                  schema_bias: Optional[np.ndarray] = None,
                  frontier_weight: float = 0.0,
                  pain_field: Optional[np.ndarray] = None,
                  membrane_field: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compose potential field from all influences.
        
        Args:
            walls_mask: Boolean mask of walls
            corner_field: Optional corner hazard field
            wall_prox_field: Optional wall proximity field
            schema_bias: Optional schema bias field
            frontier_weight: Weight for blending frontier into novelty
            pain_field: Optional pain-based repulsive field
            membrane_field: Optional protective membrane field
            
        Returns:
            Composed potential field
        """
        # Optionally blend frontier into novelty (runner's behavior today)
        novelty = self.fields["Novel"]
        if frontier_weight != 0.0:
            novelty = novelty + frontier_weight * self.fields["Frontier"]
        
        # Prepare attractors and repulsors - dynamically classify based on valence
        attractors = {}
        repulsors = {"Trail": self.fields["Trail"]}
        
        # A is usually attractive (unless learned otherwise)
        if self.valence.get("A", 1.0) >= 0:
            attractors["A"] = self.fields["A"]
        else:
            repulsors["A"] = self.fields["A"]
            
        # B can become repulsive when learned to be negative
        if self.valence.get("B", 1.0) >= 0:
            attractors["B"] = self.fields["B"]
        else:
            repulsors["B"] = self.fields["B"]
            
        # Novelty is always attractive
        attractors["Novel"] = novelty
        
        if corner_field is not None:
            repulsors["Corner"] = corner_field
        
        if wall_prox_field is not None:
            repulsors["WallProx"] = wall_prox_field
        
        if pain_field is not None:
            repulsors["Pain"] = pain_field
        
        if membrane_field is not None:
            repulsors["Membrane"] = membrane_field
            
        # Get weights based on classification
        w_attr = {}
        w_rep = {"Trail": self.cfg.w_trail}
        
        # Set weights based on whether fields are attractors or repulsors
        if "A" in attractors:
            w_attr["A"] = self.valence.get("A", 1.0)
        elif "A" in repulsors:
            w_rep["A"] = abs(self.valence.get("A", 1.0))
            
        if "B" in attractors:
            w_attr["B"] = self.valence.get("B", 1.0)
        elif "B" in repulsors:
            w_rep["B"] = abs(self.valence.get("B", 1.0))
            
        w_attr["Novel"] = self.valence.get("Novel", self.cfg.w_novel)
        
        # Add other repulsor weights
        if corner_field is not None:
            w_rep["Corner"] = self.cfg.w_corner
        if wall_prox_field is not None:
            w_rep["WallProx"] = getattr(self.cfg, "w_wall_prox", 0.3)
        if pain_field is not None:
            w_rep["Pain"] = getattr(self.cfg, "w_pain", 0.7)
        if membrane_field is not None:
            w_rep["Membrane"] = getattr(self.cfg, "w_membrane", 0.6)
        
        # Determine semiring mode based on pain  
        mode = "linear"  # default
        if hasattr(self, 'affect_state') and self.affect_state is not None:
            if self.affect_state.pain > getattr(self.cfg, 'pain_semiring_threshold', 0.6):
                mode = "maxplus"  # Use max-plus semiring under high pain
        
        # Compose potential
        return compose_potential(attractors, repulsors, w_attr, w_rep, bias=schema_bias, mode=mode)

    def compose_value(self, walls_mask: Optional[np.ndarray] = None,
                      corner_field: Optional[np.ndarray] = None,
                      wall_prox_field: Optional[np.ndarray] = None,
                      schema_bias: Optional[np.ndarray] = None,
                      frontier_weight: float = 0.0,
                      pain_field: Optional[np.ndarray] = None,
                      membrane_field: Optional[np.ndarray] = None,
                      lam: Optional[float] = None,
                      sweeps: Optional[int] = None) -> np.ndarray:
        """
        LMDP path: assemble state costs q and reward injection R_inj, run
        warm-started value sweeps, return the value field V.

        Units: everything is reward-per-step. Repulsors are COSTS the
        planner routes around (not negative attractors): q collects step
        effort, trail, hazards, membranes, pain, and negative-valence
        targets. R_inj collects positive-valence expected rewards plus the
        novelty bonus; the belief prior gives a small optimism floor
        everywhere ("unexplored cells might hold targets at prior rate").

        Planning runs on the agent's OWN map (known_walls), never the
        environment's truth; unknown space is optimistically passable.
        """
        if lam is None:
            # One source of truth: affect sets lambda for BOTH the value
            # sweeps and the action softmax (runner reads lam_current).
            affect = getattr(self, "affect_state", None)
            if affect is not None:
                from ..core.affect import affect_to_lambda
                lam = affect_to_lambda(
                    affect,
                    lam_base=self.lam,
                    k_pain=float(getattr(self.cfg, "k_pain_lambda", 0.9)),
                    k_arousal=float(getattr(self.cfg, "k_arousal_lambda", 0.3)),
                    lam_min=float(getattr(self.cfg, "lam_min", 0.005)),
                    lam_max=float(getattr(self.cfg, "lam_max", 0.1)),
                )
            else:
                lam = self.lam
        else:
            lam = float(lam)
        self.lam_current = lam

        sweeps = self.z_sweeps if sweeps is None else int(sweeps)
        fresh = self._fresh_value
        if fresh:
            # Orient before moving: converge the cold-started field once.
            extra = int(getattr(self.cfg, "init_sweeps", 0)) or (self.H + self.W)
            sweeps = sweeps + extra
            self._fresh_value = False

        # Exact barrier: membrane at/above threshold means q = +inf there,
        # implemented by excluding those cells from value propagation
        # entirely (V = -VBIG). Softmax probability of entering is exactly 0
        # whenever any non-forbidden neighbor exists.
        walls = self.known_walls
        if membrane_field is not None:
            forbidden = membrane_field >= float(getattr(self.cfg, "barrier_threshold", 0.75))
            if forbidden.any():
                walls = walls | forbidden

        # --- state costs q(v) >= 0, reward units per step ---
        cfg = self.cfg
        q = np.full((self.H, self.W), float(getattr(cfg, "q_step", 0.01)),
                    dtype=np.float32)
        q += float(getattr(cfg, "q_trail", 0.08)) * self.fields["Trail"]
        if corner_field is not None:
            q += float(getattr(cfg, "q_corner", 0.02)) * corner_field
        if wall_prox_field is not None:
            q += float(getattr(cfg, "q_wall_prox", 0.02)) * wall_prox_field
        if pain_field is not None:
            q += float(getattr(cfg, "q_pain", 0.3)) * pain_field
        if membrane_field is not None:
            q += float(getattr(cfg, "q_membrane", 0.3)) * membrane_field

        # --- reward injection R_inj(v): expected reward for arriving at v ---
        R = np.zeros((self.H, self.W), dtype=np.float32)
        for channel in ("A", "B"):
            val = float(self.valence.get(channel, 0.0))
            if val >= 0.0:
                R += val * self.fields[channel]
            else:
                # Aversive targets are state costs, not negative rewards
                q += (-val) * self.fields[channel]

        # Epistemic pull. Two regimes:
        # - infogain (default): expected information gain from belief
        #   entropy + map uncertainty, one principled term, affect-modulated
        #   (curiosity raises beta, fear lowers it).
        # - frontier fallback: diffused unseen space.
        # Self-deposited novelty -- prediction error injected AT the agent's
        # own trailing cells -- must NOT enter as an absorbing reward: the
        # agent would plan back to where it just was (a self-trap). Novel
        # still feeds affect/surprise upstream.
        mode = str(getattr(self.cfg, "epistemic_mode", "infogain"))
        if mode == "infogain" and not self.use_beliefs:
            mode = "frontier"  # infogain needs belief fields
        epistemic = np.zeros((self.H, self.W), dtype=np.float32)
        if mode == "infogain":
            from ..core.infogain import epistemic_beta, pooled_gain, uncertainty_map
            affect = getattr(self, "affect_state", None)
            beta = epistemic_beta(
                float(getattr(self.cfg, "beta_epist", 0.3)),
                arousal=affect.arousal if affect is not None else 0.0,
                pain=affect.pain if affect is not None else 0.0,
                k_curiosity=float(getattr(self.cfg, "k_curiosity", 0.5)),
                k_fear=float(getattr(self.cfg, "k_fear", 0.8)),
            )
            if beta > 0.0:
                u = uncertainty_map(self.L["A"], self.L["B"], self.seen,
                                    self.known_walls,
                                    w_map=float(getattr(self.cfg, "w_map_uncertainty", 1.0)))
                epistemic = beta * pooled_gain(u, self.win)
                R += epistemic
        elif mode == "frontier":
            epistemic = (float(self.valence.get("Novel", self.cfg.w_novel))
                         * self.fields["Frontier"]).astype(np.float32)
            R += epistemic
        # mode == "none": pure exploitation
        self.last_epistemic = epistemic  # exposed for visualization

        if schema_bias is not None:
            R += np.maximum(schema_bias, 0.0)
            q += np.maximum(-schema_bias, 0.0)

        R_inj = np.where(R > 1e-6, R, -VBIG).astype(np.float32)

        if int(getattr(self.cfg, "pyramid_levels", 1)) >= 2 and fresh:
            # Coarse-to-fine COLD-START acceleration only. Measured: as an
            # every-tick lower bound the coarse level injects optimism bias
            # (its pooling opens wall gaps that are closed at fine scale)
            # and slightly hurts behavior; as a one-shot initializer it
            # provably speeds convergence (tests/test_pyramid.py).
            from ..core.pyramid import pyramid_value_sweeps
            self.V, self.last_residuals, self.V_coarse = pyramid_value_sweeps(
                self.V, q, R_inj, walls, lam=lam, sweeps=sweeps,
                V_coarse=getattr(self, "V_coarse", None)
            )
        else:
            self.V, self.last_residuals = value_sweeps(
                self.V, q, R_inj, walls, lam=lam, sweeps=sweeps
            )
        # Expose sweep inputs for offline diagnostics (fixed-point deep
        # verification in scripts/exp_kappa.py) -- references, not copies.
        self.last_q = q
        self.last_R_inj = R_inj
        self.last_walls_used = walls
        return self.V

    def step(self, obs_vec: np.ndarray) -> Tuple[int, Dict[str, np.ndarray]]:
        """
        Process observation and update internal fields.
        Compatible with ChemotaxisAgentCA interface for drop-in replacement.
        
        Args:
            obs_vec: Flattened observation vector
            
        Returns:
            Tuple of (action=-1, fields dictionary)
        """
        self.step_fields(obs_vec)
        return -1, self.get_fields()
    
    def get_fields(self) -> Dict[str, np.ndarray]:
        """
        Get all current fields for visualization/debugging.
        
        Returns:
            Dictionary of field copies
        """
        return {
            "GA": self.fields["A"].copy(),
            "GB": self.fields["B"].copy(),
            "Vtrail": self.fields["Trail"].copy(),
            "Novel": self.fields["Novel"].copy(),
            "Frontier": self.fields["Frontier"].copy(),
            "known_walls": self.known_walls.copy(),
        }