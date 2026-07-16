"""ForageWorld environment implementation."""

from typing import Dict, Tuple

import numpy as np

from ..configs import EnvConfig


class ForageWorld:
    """
    Grid world with random walls and scattered targets A and B.
    
    Observation is local patch (win x win) with 4 channels: walls, A, B, agent.
    """
    
    EMPTY, WALL, A, B = 0, 1, 2, 3

    def __init__(self, cfg: EnvConfig):
        """
        Initialize ForageWorld environment.
        
        Args:
            cfg: Environment configuration
        """
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.seed)
        self.H, self.W = cfg.H, cfg.W
        self.win = cfg.win
        self.max_steps = cfg.max_steps

        self.grid = np.zeros((self.H, self.W), dtype=np.int32)
        self.walls = np.zeros_like(self.grid, dtype=bool)
        self.TA = np.zeros_like(self.grid, dtype=bool)
        self.TB = np.zeros_like(self.grid, dtype=bool)

        self.t = 0
        self.y = 0
        self.x = 0
        self.reward_A = float(cfg.reward_A)
        self.reward_B = float(cfg.reward_B)
        self._respawn_queue = []

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation vector
        """
        self.grid[:] = self.EMPTY
        
        # Generate walls
        self.walls = (self.rng.rand(self.H, self.W) < self.cfg.p_wall)
        # Keep a 1-cell border free to reduce traps
        self.walls[0,:] = self.walls[-1,:] = False
        self.walls[:,0] = self.walls[:,-1] = False

        # Place targets
        self.TA[:] = False
        self.TB[:] = False
        free = np.where(~self.walls)
        idx = self.rng.choice(
            free[0].size, 
            size=min(self.cfg.n_targets_A + self.cfg.n_targets_B, free[0].size), 
            replace=False
        )
        coords = list(zip(free[0][idx], free[1][idx]))
        
        for i, (yy, xx) in enumerate(coords[:self.cfg.n_targets_A]):
            self.TA[yy, xx] = True
        for i, (yy, xx) in enumerate(coords[self.cfg.n_targets_A:self.cfg.n_targets_A + self.cfg.n_targets_B]):
            self.TB[yy, xx] = True

        # Spawn agent at random free cell not on a target
        free2 = np.where(~self.walls & ~self.TA & ~self.TB)
        if free2[0].size == 0:
            # Extremely rare with high p_wall; fallback
            self.walls[:] = False
            return self.reset()
            
        i = self.rng.choice(free2[0].size)
        self.y, self.x = int(free2[0][i]), int(free2[1][i])

        self.t = 0
        # Dynamic reward values (the swap event flips them mid-episode);
        # step() and learners must read these, not cfg.
        self.reward_A = float(self.cfg.reward_A)
        self.reward_B = float(self.cfg.reward_B)
        # Pending respawns: list of [delay_remaining, kind]
        self._respawn_queue = []
        return self._obs()

    def _random_free_cell(self):
        free = np.where(~self.walls & ~self.TA & ~self.TB)
        if free[0].size == 0:
            return None
        i = self.rng.choice(free[0].size)
        yy, xx = int(free[0][i]), int(free[1][i])
        if (yy, xx) == (self.y, self.x):
            return None  # skip this tick rather than spawn under the agent
        return yy, xx

    def _apply_nonstationarity(self):
        """Regrow / drift / swap events. All randomness from self.rng, so
        schedules are seed-deterministic."""
        # regrow: tick down pending respawns
        if self._respawn_queue:
            still = []
            for item in self._respawn_queue:
                item[0] -= 1
                if item[0] <= 0:
                    cell = self._random_free_cell()
                    if cell is None:
                        item[0] = 1  # try again next tick
                        still.append(item)
                        continue
                    (self.TA if item[1] == "A" else self.TB)[cell] = True
                else:
                    still.append(item)
            self._respawn_queue = still

        # drift: periodic teleport within a Chebyshev ball
        if self.cfg.T_shift > 0 and self.t % self.cfg.T_shift == 0:
            for grid in (self.TA, self.TB):
                for (yy, xx) in np.argwhere(grid):
                    if self.rng.rand() >= self.cfg.p_move:
                        continue
                    r = self.cfg.r_drift
                    y0, y1 = max(0, yy - r), min(self.H, yy + r + 1)
                    x0, x1 = max(0, xx - r), min(self.W, xx + r + 1)
                    region_free = (~self.walls & ~self.TA & ~self.TB)
                    cand = np.argwhere(region_free[y0:y1, x0:x1])
                    cand = [(y0 + cy, x0 + cx) for cy, cx in cand
                            if (y0 + cy, x0 + cx) != (self.y, self.x)]
                    if cand:
                        ny, nx = cand[self.rng.choice(len(cand))]
                        grid[yy, xx] = False
                        grid[ny, nx] = True

        # swap: one-time reward revaluation
        if self.cfg.T_swap > 0 and self.t == self.cfg.T_swap:
            self.reward_A, self.reward_B = self.reward_B, self.reward_A

    def clone(self) -> "ForageWorld":
        """Deep copy of the full world state INCLUDING rng state, so a
        clairvoyant reference policy can run the same stochastic world."""
        other = ForageWorld(self.cfg)
        other.grid = self.grid.copy()
        other.walls = self.walls.copy()
        other.TA = self.TA.copy()
        other.TB = self.TB.copy()
        other.t = self.t
        other.y, other.x = self.y, self.x
        other.reward_A = getattr(self, "reward_A", float(self.cfg.reward_A))
        other.reward_B = getattr(self, "reward_B", float(self.cfg.reward_B))
        other._respawn_queue = [list(item) for item in
                                getattr(self, "_respawn_queue", [])]
        other.rng = np.random.RandomState()
        other.rng.set_state(self.rng.get_state())
        return other

    def _obs(self) -> np.ndarray:
        """
        Generate observation vector.
        
        Returns:
            Flattened observation array
        """
        half = self.win // 2
        ch = 4
        patch = np.zeros((ch, self.win, self.win), dtype=np.float32)
        
        for dy in range(-half, half+1):
            for dx in range(-half, half+1):
                yy, xx = self.y + dy, self.x + dx
                py, px = dy + half, dx + half
                if 0 <= yy < self.H and 0 <= xx < self.W:
                    patch[0, py, px] = 1.0 if self.walls[yy, xx] else 0.0
                    patch[1, py, px] = 1.0 if self.TA[yy, xx] else 0.0
                    patch[2, py, px] = 1.0 if self.TB[yy, xx] else 0.0
                    patch[3, py, px] = 1.0 if (yy == self.y and xx == self.x) else 0.0
                else:
                    # Out-of-bounds is impassable: report it as wall. An
                    # agent without a priori knowledge of the world size
                    # must be able to DISCOVER the boundary by looking.
                    patch[0, py, px] = 1.0

        return patch.reshape(-1)

    def step(self, a: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action in environment.
        
        Args:
            a: Action index (0=up, 1=down, 2=left, 3=right)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.t += 1
        self._apply_nonstationarity()
        dy, dx = [(-1,0),(1,0),(0,-1),(0,1)][int(a)]
        ny, nx = self.y + dy, self.x + dx
        reward = self.cfg.step_cost
        moved = False
        picked = None

        if (0 <= ny < self.H) and (0 <= nx < self.W) and (not self.walls[ny, nx]):
            # Odometry slip (off by default): the move succeeds but lands on
            # a different passable neighbor. info["moved"] stays True -- the
            # proprioceptive lie an egocentric agent must detect and correct.
            p_slip = float(getattr(self.cfg, "p_slip", 0.0))
            if p_slip > 0.0 and self.rng.rand() < p_slip:
                alts = []
                for ddy, ddx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ay, ax = self.y + ddy, self.x + ddx
                    if (0 <= ay < self.H and 0 <= ax < self.W
                            and not self.walls[ay, ax] and (ay, ax) != (ny, nx)):
                        alts.append((ay, ax))
                if alts:
                    ny, nx = alts[self.rng.choice(len(alts))]
            self.y, self.x = ny, nx
            moved = True
        else:
            reward += self.cfg.bump_pen

        # Pickup target (dynamic reward values: swap can flip them mid-episode)
        if self.TA[self.y, self.x]:
            reward += self.reward_A
            self.TA[self.y, self.x] = False
            picked = "A"
        elif self.TB[self.y, self.x]:
            reward += self.reward_B
            self.TB[self.y, self.x] = False
            picked = "B"

        if picked is not None and self.cfg.p_regrow > 0.0:
            # Geometric respawn delay, seed-deterministic
            delay = int(self.rng.geometric(self.cfg.p_regrow))
            self._respawn_queue.append([delay, picked])

        regrow_on = self.cfg.p_regrow > 0.0
        collected_all = (not self.TA.any() and not self.TB.any()
                         and not self._respawn_queue)
        done = (self.t >= self.max_steps) or (collected_all and not regrow_on)
        info = {"moved": moved, "picked": picked}

        return self._obs(), float(reward), bool(done), info

    def render_rgb(self) -> np.ndarray:
        """
        Render environment as RGB image.
        
        Returns:
            RGB image array (H, W, 3) with values in [0, 255]
        """
        rgb = np.ones((self.H, self.W, 3), dtype=np.float32) * 0.95
        rgb[self.walls] = np.array([0.40,0.40,0.40], dtype=np.float32)
        
        # Targets
        rgb[self.TA & (~self.walls)] = np.array([0.45,0.85,0.55], dtype=np.float32)  # green
        rgb[self.TB & (~self.walls)] = np.array([0.85,0.55,0.85], dtype=np.float32)  # magenta
        
        # Agent
        rgb[self.y, self.x] = np.array([0.05,0.35,0.95], dtype=np.float32)
        
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)