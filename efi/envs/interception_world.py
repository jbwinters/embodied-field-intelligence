"""Local-sensing interception in rooms, with optional interior obstacles.

The target moves synchronously by one cell, reflecting at walls. Collection
requires sharing the arrival cell (exchanging cells does not collect). It is
an externally moving object, independent of the agent's actions. Its heading,
track and motion law never enter the observation.
"""

from dataclasses import dataclass

import numpy as np

from ..core.anticipation import MOTIONS


@dataclass
class InterceptionConfig:
    H: int = 9
    W: int = 13
    win: int = 5
    max_steps: int = 24
    seed: int = 0
    rotate: int = 0
    obstacles: bool = False
    hazards: bool = False
    step_cost: float = -0.01
    goal_reward: float = 1.0
    collision_cost: float = -2.0

    def __post_init__(self):
        if min(self.H, self.W) < 9 or self.H % 2 == 0 or self.W % 2 == 0:
            raise ValueError("odd dimensions >= 9 required")
        if self.win < 3 or self.win % 2 == 0 or self.max_steps < 1:
            raise ValueError("odd window >= 3 and positive step allowance required")


class InterceptionWorld:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.seed)
        self.win = cfg.win
        self.max_steps = cfg.max_steps

    def reset(self):
        cfg = self.cfg
        walls = np.zeros((cfg.H, cfg.W), dtype=bool)
        walls[[0, -1], :] = True
        walls[:, [0, -1]] = True
        cy, cx = cfg.H // 2, cfg.W // 2
        if cfg.obstacles:
            walls[cy, cx - 1 : cx + 2] = True
            walls[cy + 2, 2:-2:3] = True
        self.walls = np.rot90(walls, cfg.rotate).copy()
        self.H, self.W = self.walls.shape

        def rotate_point(y, x):
            marker = np.zeros(walls.shape, dtype=bool)
            marker[y, x] = True
            return tuple(int(v) for v in np.argwhere(np.rot90(marker, cfg.rotate))[0])

        self.y, self.x = rotate_point(cy + 1, cx)
        self.track = [rotate_point(cy - 1, x) for x in range(1, cfg.W - 1)]
        self.target_index = cx - 1 + int(self.rng.choice([-2, -1, 1, 2]))
        self.direction = int(self.rng.choice([-1, 1]))
        self.hazard_track = [rotate_point(y, cx + 2) for y in range(1, cfg.H - 1)]
        if cfg.obstacles:
            # Keep both externally moving tracks clear. Interior obstacles
            # constrain the agent's routes, not the supplied motion law.
            for point in self.hazard_track:
                self.walls[point] = False
        self.hazard_index = cy - 1 + int(self.rng.choice([-1, 0, 1]))
        self.hazard_direction = int(self.rng.choice([-1, 1]))
        self.t = 0
        self.success = False
        return self.observation()

    @property
    def target(self):
        return self.track[self.target_index]

    @property
    def hazard(self):
        return self.hazard_track[self.hazard_index] if self.cfg.hazards else None

    def observation(self):
        patch = np.zeros((6, self.win, self.win), dtype=np.float32)
        half = self.win // 2
        for iy in range(self.win):
            for ix in range(self.win):
                y, x = self.y + iy - half, self.x + ix - half
                if not (0 <= y < self.H and 0 <= x < self.W):
                    patch[0, iy, ix] = 1
                    continue
                patch[0, iy, ix] = self.walls[y, x]
                patch[3, iy, ix] = (y, x) == (self.y, self.x)
                patch[5, iy, ix] = (y, x) == self.target and not self.success
                patch[4, iy, ix] = (y, x) == self.hazard
        return patch.reshape(-1)

    def step(self, action):
        if action not in range(5):
            raise ValueError("action must be a move (0..3) or wait (4)")
        self.t += 1
        old_hazard = self.hazard
        if self.cfg.hazards:
            next_hazard = self.hazard_index + self.hazard_direction
            if not 0 <= next_hazard < len(self.hazard_track):
                self.hazard_direction *= -1
                next_hazard = self.hazard_index + self.hazard_direction
            self.hazard_index = next_hazard
        new_index = self.target_index + self.direction
        if not 0 <= new_index < len(self.track):
            self.direction *= -1
            new_index = self.target_index + self.direction
        self.target_index = new_index
        old_pos = self.y, self.x
        dy, dx = MOTIONS[action]
        y, x = self.y + dy, self.x + dx
        if 0 <= y < self.H and 0 <= x < self.W and not self.walls[y, x]:
            self.y, self.x = y, x
        moved = (self.y, self.x) != old_pos
        collision = self.cfg.hazards and (
            (self.y, self.x) == self.hazard
            or ((self.y, self.x) == old_hazard and old_pos == self.hazard)
        )
        self.success = (self.y, self.x) == self.target and not collision
        reward = self.cfg.step_cost + self.cfg.goal_reward * self.success
        reward += self.cfg.collision_cost * collision
        done = self.success or collision or self.t >= self.max_steps
        return (
            self.observation(),
            float(reward),
            done,
            {
                "moved": moved,
                "picked": "A" if self.success else None,
                "success": self.success,
                "collision": collision,
                "bump": action != 4 and not moved,
                "wait": action == 4,
            },
        )
