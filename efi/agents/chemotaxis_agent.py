"""Chemotaxis agent with CA-based navigation."""

from typing import Dict, Tuple

import numpy as np

from ..configs import AgentConfig, Ablations
from ..core import diffuse_masked, update_visit_trail, update_novelty, compute_reachable_frontier, wall_proximity_field


class ChemotaxisAgentCA:
    """
    Chemotaxis agent using cellular automata fields for navigation.
    
    The agent maintains multiple fields:
    - GA, GB: Scent fields for targets A and B
    - V: Visit trail (repulsive)
    - Nv: Novelty field (attractive)
    - known_walls: Discovered wall locations
    """
    
    def __init__(self, env, cfg: AgentConfig, ablate: Ablations):
        """
        Initialize chemotaxis agent.
        
        Args:
            env: ForageWorld environment
            cfg: Agent configuration
            ablate: Ablation flags
        """
        self.env = env
        self.cfg = cfg
        self.ablate = ablate
        self.rng = np.random.RandomState(cfg.seed)
        self.H, self.W, self.win = env.H, env.W, env.win
        # Initialize valence weights as dictionary (persist across episodes for learning)
        self.valence = {
            "A": float(self.cfg.valA_init),
            "B": float(self.cfg.valB_init),
            "Novel": float(self.cfg.w_novel),  # Can be learned slowly if desired
        }
        # Backwards compatibility
        self.valA = self.valence["A"]
        self.valB = self.valence["B"]
        self.reset()

    def learn_valence(self, channel: str, reward: float):
        """Update valence weights based on experienced reward."""
        lr = float(self.cfg.valence_lr)
        clip = float(getattr(self.cfg, "valence_clip", 1.5))
        self.valence[channel] = float(np.clip(self.valence.get(channel, 0.0) + lr * reward, -clip, clip))
        # Update backwards compatibility attributes
        if channel == "A":
            self.valA = self.valence["A"]
        elif channel == "B":
            self.valB = self.valence["B"]
    
    def learn_valence_counterfactual(self, field_values_at_action: dict, field_values_alternatives: dict, reward: float):
        """
        Update valence weights using counterfactual credit assignment.
        
        Args:
            field_values_at_action: Field values at the chosen action location
            field_values_alternatives: Average field values at alternative actions
            reward: The immediate reward received
        """
        lr_step = float(getattr(self.cfg, "valence_lr_step", 0.001))  # Small learning rate for per-step updates
        clip = float(getattr(self.cfg, "valence_clip", 1.5))
        
        for channel in ["A", "B", "Novel"]:
            if channel in field_values_at_action and channel in field_values_alternatives:
                # Counterfactual gradient: advantage of chosen action over alternatives
                advantage = field_values_at_action[channel] - field_values_alternatives[channel]
                delta = lr_step * reward * advantage
                
                old_val = self.valence.get(channel, 0.0)
                self.valence[channel] = float(np.clip(old_val + delta, -clip, clip))
        
        # Update backwards compatibility
        self.valA = self.valence["A"]
        self.valB = self.valence["B"]
    
    def reset(self):
        """Reset agent state for new episode."""
        self.GA = np.zeros((self.H, self.W), dtype=np.float32)
        self.GB = np.zeros((self.H, self.W), dtype=np.float32)
        self.V  = np.zeros((self.H, self.W), dtype=np.float32)
        self.Nv = np.zeros((self.H, self.W), dtype=np.float32)
        self.known_walls = np.zeros((self.H, self.W), dtype=bool)
        self.seen = np.zeros((self.H, self.W), dtype=bool)  # for frontier drive
        # Mark initial visible area as seen
        y, x = self.env.y, self.env.x
        half = self.win // 2
        for dy in range(-half, half+1):
            for dx in range(-half, half+1):
                gy, gx = y + dy, x + dx
                if 0 <= gy < self.H and 0 <= gx < self.W:
                    self.seen[gy, gx] = True
        self.prev_P_here = 0.0
        self.stuck_count = 0
        self.last_pos = (self.env.y, self.env.x)
        self.last_action = None
        self._prev_patch = None  # for obs-change novelty
        from collections import deque
        self._pos_hist = deque(maxlen=3)
        self._pos_hist.append((self.env.y, self.env.x))


    def step(self, obs_vec: np.ndarray) -> Tuple[int, Dict[str, np.ndarray]]:
        """
        Process observation and update internal fields.
        Action is chosen externally from the runner based on fields.
        """
        y, x = self.env.y, self.env.x
        
        # Ping-pong detection
        self._pos_hist.append((y, x))
        pingpong = (len(self._pos_hist) == 3 and 
                   self._pos_hist[0] == self._pos_hist[2] and 
                   self._pos_hist[0] != self._pos_hist[1])

        # 1) Seed scents locally from patch
        ch = 4
        patch = obs_vec[:ch*self.win*self.win].reshape(ch, self.win, self.win)
        walls_local = (patch[0] > 0.5)
        A_local     = (patch[1] > 0.5)
        B_local     = (patch[2] > 0.5)
        half = self.win // 2

        for dy in range(-half, half+1):
            for dx in range(-half, half+1):
                gy, gx = y + dy, x + dx
                py, px = dy + half, dx + half
                if 0 <= gy < self.H and 0 <= gx < self.W:
                    self.seen[gy, gx] = True  # Mark as seen for frontier
                    if walls_local[py, px]:
                        self.known_walls[gy, gx] = True
                    if A_local[py, px]:
                        self.GA[gy, gx] = max(self.GA[gy, gx], self.cfg.seed_strength)
                    if B_local[py, px]:
                        self.GB[gy, gx] = max(self.GB[gy, gx], self.cfg.seed_strength)

        passable_mask = ~self.known_walls
        walls_mask = ~passable_mask  # only *known* walls block; discovery grows over time

        # 2) Diffuse GA/GB with wall blocking
        self.GA = diffuse_masked(self.GA, walls_mask, diff=self.cfg.scent_diff,
                                 decay=self.cfg.scent_decay, steps=self.cfg.scent_steps)
        self.GB = diffuse_masked(self.GB, walls_mask, diff=self.cfg.scent_diff,
                                 decay=self.cfg.scent_decay, steps=self.cfg.scent_steps)

        # Optional internal thinking ticks
        for _ in range(max(0, int(self.cfg.internal_think))):
            self.GA = diffuse_masked(self.GA, walls_mask, diff=self.cfg.scent_diff, decay=self.cfg.scent_decay, steps=1)
            self.GB = diffuse_masked(self.GB, walls_mask, diff=self.cfg.scent_diff, decay=self.cfg.scent_decay, steps=1)

        # 3) Trail (repulsive) with boost for ping-ponging
        if self.ablate.trail:
            # Stronger injection when ping-ponging
            v_inj_eff = self.cfg.v_inj * (2.5 if pingpong else 1.0)
                
            self.V = update_visit_trail(self.V, y, x, walls_mask,
                                        v_decay=self.cfg.v_decay,
                                        v_diff=self.cfg.v_diff,
                                        v_inj=v_inj_eff)
        else:
            self.V[:] = 0.0

        # 4) Novelty = 0.5*Δ(scent at here) + 0.5*obs-change in (walls,A,B) patch
        P_local = float(self.GA[y, x] + self.GB[y, x])
        d_scent = abs(P_local - self.prev_P_here)
        self.prev_P_here = P_local

        d_obs = 0.0
        if self._prev_patch is not None:
            # exclude the agent channel (index 3)
            d_obs = float(np.mean(np.abs(patch[:3] - self._prev_patch[:3])))
        self._prev_patch = patch.copy()

        pred_err = 0.5 * d_scent + 0.5 * d_obs

        if self.ablate.novelty:
            self.Nv = update_novelty(self.Nv, pred_err, y, x, walls_mask, n_decay=0.018, n_diff=0.06, gain=6.0)
        else:
            self.Nv[:] = 0.0

        # 5) Frontier field (reachability-aware)
        # Use flood-fill to only consider frontiers reachable from current position
        if getattr(self.cfg, 'reachable_frontier', True):
            U = compute_reachable_frontier(self.seen, self.known_walls, y, x)
        else:
            # Legacy behavior - simple unseen field
            U = (~self.seen).astype(np.float32)   # 1 for unknown cells
            U = U * (~walls_mask).astype(np.float32)
            U = diffuse_masked(U, walls_mask, diff=0.15, decay=0.01, steps=3)

        # NOTE: we do NOT update stuck_count here; runner updates it *after* env.step()
        
        # Compute wall proximity for visualization
        W_prox = wall_proximity_field(self.known_walls, radius=getattr(self.cfg, "wall_prox_radius", 1.5))

        fields = {
            "GA": self.GA.copy(),
            "GB": self.GB.copy(),
            "Vtrail": self.V.copy(),
            "Novel": self.Nv.copy(),
            "Frontier": U.copy(),
            "known_walls": self.known_walls.copy(),
            "WallProx": W_prox.copy(),
        }
        return -1, fields
    
    def compose_P(self, walls_mask: np.ndarray,
                  corner_field: np.ndarray = None,
                  wall_prox_field: np.ndarray = None,
                  schema_bias: np.ndarray = None,
                  frontier_weight: float = 0.0) -> np.ndarray:
        """
        Compose potential field from all influences.
        Provides same interface as FieldController for consistency.
        
        Args:
            walls_mask: Boolean mask of walls
            corner_field: Optional corner hazard field
            schema_bias: Optional schema bias field
            frontier_weight: Weight for blending frontier into novelty
            
        Returns:
            Composed potential field
        """
        from ..core.potential import compose_potential
        
        # Blend frontier into novelty if requested
        novelty = self.Nv
        if frontier_weight != 0.0 and hasattr(self, 'Frontier'):
            novelty = novelty + frontier_weight * getattr(self, 'Frontier', np.zeros_like(novelty))
        
        # Prepare fields
        attractors = {
            "A": self.GA,
            "B": self.GB,
            "Novel": novelty,
        }
        
        repulsors = {"Trail": self.V}
        if corner_field is not None:
            repulsors["Corner"] = corner_field
        if wall_prox_field is not None:
            repulsors["WallProx"] = wall_prox_field
        
        # Get weights from valence dict
        w_attr = {
            "A": self.valence.get("A", 1.0),
            "B": self.valence.get("B", 1.0),
            "Novel": self.valence.get("Novel", self.cfg.w_novel),
        }
        
        w_rep = {
            "Trail": self.cfg.w_trail,
            "Corner": self.cfg.w_corner if corner_field is not None else 0.0,
            "WallProx": getattr(self.cfg, "w_wall_prox", 0.3) if wall_prox_field is not None else 0.0,
        }
        
        # Compose
        return compose_potential(attractors, repulsors, w_attr, w_rep, bias=schema_bias)