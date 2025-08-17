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
        return self._obs()

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
        dy, dx = [(-1,0),(1,0),(0,-1),(0,1)][int(a)]
        ny, nx = self.y + dy, self.x + dx
        reward = self.cfg.step_cost
        moved = False
        picked = None
        
        if (0 <= ny < self.H) and (0 <= nx < self.W) and (not self.walls[ny, nx]):
            self.y, self.x = ny, nx
            moved = True
        else:
            reward += self.cfg.bump_pen

        # Pickup target
        if self.TA[self.y, self.x]:
            reward += self.cfg.reward_A
            self.TA[self.y, self.x] = False
            picked = "A"
        elif self.TB[self.y, self.x]:
            reward += self.cfg.reward_B
            self.TB[self.y, self.x] = False
            picked = "B"

        done = (self.t >= self.max_steps) or (not self.TA.any() and not self.TB.any())
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