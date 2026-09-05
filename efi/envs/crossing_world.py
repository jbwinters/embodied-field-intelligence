"""A local-sensing crossing task with a moving, noncollectible hazard.

World dynamics alone know hazard velocity. A hazard moves one cell per
tick, reflecting at the lane ends. In reverse mode it instead reverses on
every tick. Agent and hazard move synchronously; sharing a destination or
exchanging cells is a collision. Waiting consumes time and is never a bump.
"""

from dataclasses import dataclass

import numpy as np

from ..core.anticipation import MOTIONS


@dataclass
class CrossingConfig:
    H: int = 9
    W: int = 9
    win: int = 5
    max_steps: int = 60
    seed: int = 0
    rule: str = "continue"
    rotate: int = 0
    switch_step: int = 0
    step_cost: float = -0.01
    collision_cost: float = -2.0
    goal_reward: float = 1.0

    def __post_init__(self):
        if self.H < 7 or self.W < 7 or self.H % 2 == 0 or self.W % 2 == 0:
            raise ValueError("crossing dimensions must be odd and at least 7")
        if self.win < 3 or self.win % 2 == 0 or self.max_steps < 1:
            raise ValueError("positive episode length and odd observation window >= 3 required")
        if self.rule not in ("continue", "reverse"):
            raise ValueError("unknown motion rule")


class CrossingWorld:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.seed)
        self.win = cfg.win
        self.max_steps = cfg.max_steps

    def reset(self):
        cfg = self.cfg
        self.H, self.W = cfg.H, cfg.W
        cy, cx = self.H // 2, self.W // 2
        walls = np.ones((self.H, self.W), dtype=bool)
        walls[cy, 1:-1] = False
        walls[1:-1, cx] = False
        # All endpoints are specified in world construction, never supplied
        # as coordinates to the controller. Goal is seen through its window.
        self.walls = np.rot90(walls, cfg.rotate).copy()
        self.H, self.W = self.walls.shape

        def rotate_point(y, x):
            marker = np.zeros((cfg.H, cfg.W), dtype=bool)
            marker[y, x] = True
            return tuple(int(v) for v in np.argwhere(np.rot90(marker, cfg.rotate))[0])

        self.y, self.x = rotate_point(cy, cx - 1)
        self.goal = rotate_point(cy, cx + 2)
        self.lane = [rotate_point(y, cx) for y in range(1, cfg.H - 1)]
        self.hazard_index = int(self.rng.choice([cy - 2, cy - 1, cy]))
        self.direction = int(self.rng.choice([-1, 1]))
        self.rule = cfg.rule
        self.t = 0
        self.success = False
        return self.observation()

    @property
    def hazard(self):
        return self.lane[self.hazard_index]

    def observation(self):
        patch = np.zeros((5, self.win, self.win), dtype=np.float32)
        half = self.win // 2
        for iy in range(self.win):
            for ix in range(self.win):
                y, x = self.y + iy - half, self.x + ix - half
                if not (0 <= y < self.H and 0 <= x < self.W):
                    patch[0, iy, ix] = 1
                    continue
                patch[0, iy, ix] = self.walls[y, x]
                patch[1, iy, ix] = (y, x) == self.goal
                patch[3, iy, ix] = (y, x) == (self.y, self.x)
                patch[4, iy, ix] = (y, x) == self.hazard
        return patch.reshape(-1)

    def step(self, action):
        if action not in range(5):
            raise ValueError("action must be a move (0..3) or wait (4)")
        self.t += 1
        if self.cfg.switch_step and self.t == self.cfg.switch_step:
            self.rule = "reverse" if self.rule == "continue" else "continue"
        old_hazard = self.hazard
        if self.rule == "reverse":
            self.direction *= -1
        new_index = self.hazard_index + self.direction
        if not 0 <= new_index < len(self.lane):
            self.direction *= -1
            new_index = self.hazard_index + self.direction
        self.hazard_index = new_index
        old_pos = self.y, self.x
        dy, dx = MOTIONS[action]
        y, x = self.y + dy, self.x + dx
        valid = 0 <= y < self.H and 0 <= x < self.W and not self.walls[y, x]
        if valid:
            self.y, self.x = y, x
        moved = (self.y, self.x) != old_pos
        collision = (self.y, self.x) == self.hazard or (
            (self.y, self.x) == old_hazard and old_pos == self.hazard
        )
        self.success = (self.y, self.x) == self.goal and not collision
        reward = self.cfg.step_cost
        reward += self.cfg.collision_cost if collision else 0
        reward += self.cfg.goal_reward if self.success else 0
        done = collision or self.success or self.t >= self.max_steps
        return (
            self.observation(),
            float(reward),
            done,
            {
                "moved": moved,
                "picked": "A" if self.success else None,
                "collision": collision,
                "success": self.success,
                "bump": action != 4 and not moved,
                "wait": action == 4,
            },
        )
