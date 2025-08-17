"""Chemotaxis agent with CA-based navigation."""

from typing import Dict, Tuple

import numpy as np

from ..configs import AgentConfig, Ablations
from ..core import diffuse_masked, update_visit_trail, update_novelty


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
        # Initialize valence weights (persist across episodes for learning)
        self.valA = float(self.cfg.valA_init)
        self.valB = float(self.cfg.valB_init)
        self.reset()

    def learn_valence(self, kind: str, reward: float):
        """Update valence weights based on experienced reward."""
        lr = float(self.cfg.valence_lr)
        clip = float(getattr(self.cfg, "valence_clip", 1.5))
        if kind == "A":
            self.valA = float(np.clip(self.valA + lr * reward, -clip, clip))
        elif kind == "B":
            self.valB = float(np.clip(self.valB + lr * reward, -clip, clip))
    
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

    def _seed_from_patch(self, obs_vec: np.ndarray, y: int, x: int):
        """
        Seed scent fields from local observation patch.
        
        Args:
            obs_vec: Flattened observation vector
            y, x: Agent position
        """
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
                    if walls_local[py, px]:
                        self.known_walls[gy, gx] = True
                    if A_local[py, px]:
                        self.GA[gy, gx] = max(self.GA[gy, gx], self.cfg.seed_strength)
                    if B_local[py, px]:
                        self.GB[gy, gx] = max(self.GB[gy, gx], self.cfg.seed_strength)

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

        # 5) Frontier field (unseen areas)
        # Only consider frontiers that are reachable (not behind walls)
        U = (~self.seen).astype(np.float32)   # 1 for unknown cells
        # Mask out walls from frontier to prevent attraction through walls
        U = U * (~walls_mask).astype(np.float32)
        # Diffuse with stronger decay to keep frontier local
        U = diffuse_masked(U, walls_mask, diff=0.15, decay=0.01, steps=3)

        # NOTE: we do NOT update stuck_count here; runner updates it *after* env.step()

        fields = {
            "GA": self.GA.copy(),
            "GB": self.GB.copy(),
            "Vtrail": self.V.copy(),
            "Novel": self.Nv.copy(),
            "Frontier": U.copy(),
            "known_walls": self.known_walls.copy(),
        }
        return -1, fields