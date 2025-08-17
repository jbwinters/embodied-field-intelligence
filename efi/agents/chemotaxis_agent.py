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
        self.reset()

    def reset(self):
        """Reset agent state for new episode."""
        self.GA = np.zeros((self.H, self.W), dtype=np.float32)
        self.GB = np.zeros((self.H, self.W), dtype=np.float32)
        self.V  = np.zeros((self.H, self.W), dtype=np.float32)
        self.Nv = np.zeros((self.H, self.W), dtype=np.float32)
        self.known_walls = np.zeros((self.H, self.W), dtype=bool)
        self.prev_P_here = 0.0
        self.stuck_count = 0
        self.last_pos = (self.env.y, self.env.x)
        self._prev_patch = None  # for obs-change novelty

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

        # 3) Trail (repulsive)
        if self.ablate.trail:
            self.V = update_visit_trail(self.V, y, x, walls_mask,
                                        v_decay=self.cfg.v_decay, v_diff=self.cfg.v_diff, v_inj=self.cfg.v_inj)
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
            self.Nv = update_novelty(self.Nv, pred_err, y, x, walls_mask, n_decay=0.018, n_diff=0.06)
        else:
            self.Nv[:] = 0.0

        # NOTE: we do NOT update stuck_count here; runner updates it *after* env.step()

        fields = {
            "GA": self.GA.copy(),
            "GB": self.GB.copy(),
            "Vtrail": self.V.copy(),
            "Novel": self.Nv.copy(),
            "known_walls": self.known_walls.copy(),
        }
        return -1, fields