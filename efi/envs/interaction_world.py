"""Small contact worlds. Hidden reaction laws live only in the environment.

Goal paint is visible beneath an occupant; reaching it requires moving the
occupant. Same-color occupants can push forward, yield left/right, or resist.
The agent receives only a local wall/goal/hazard/body/occupant window and
actual displacement. No law, coordinates, wall code, or task label is sent.
"""

from dataclasses import dataclass

import numpy as np

from ..core.anticipation import MOTIONS


@dataclass(frozen=True)
class InteractionWorldConfig:
    seed: int = 0
    rule: str = "push"
    rotate: int = 0
    size: int = 9
    context: int = 0
    acquisition: bool = False
    layout: str = "west"
    max_steps: int = 2

    def __post_init__(self):
        if self.rule not in ("push", "left", "right", "fixed"):
            raise ValueError("unknown contact rule")
        if self.size < 9 or self.size % 2 != 1 or self.max_steps < 1:
            raise ValueError("odd room >=9 and positive episode length required")
        if self.layout not in ("west", "north", "detour"):
            raise ValueError("unknown local arrangement")


class InteractionWorld:
    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        cfg = self.cfg
        self.H = self.W = cfg.size
        self.walls = np.zeros((cfg.size, cfg.size), dtype=bool)
        self.walls[[0, -1], :] = True
        self.walls[:, [0, -1]] = True
        self.goals = np.zeros_like(self.walls, dtype=np.float32)
        self.hazards = np.zeros_like(self.goals)
        c = cfg.size // 2
        self.body = (c, c)
        self.occupant = (c - 1, c) if cfg.acquisition else (c - 1, c - 1)
        if cfg.acquisition:
            for d, (dy, dx) in enumerate(MOTIONS[:4]):
                if cfg.context & (1 << d):
                    self.walls[self.occupant[0] + dy, self.occupant[1] + dx] = True
            self.walls[self.body] = False
        elif cfg.layout == "detour":
            self.occupant = (c - 1, c)
            self.goals[c - 1, c - 1] = 1.0
            self.walls[c - 2, c] = True
        else:
            self.goals[self.occupant] = 1.0
            self.walls[(c - 1, c - 2) if cfg.layout == "west" else (c - 2, c - 1)] = True
        # Extra room geometry is outside the local interaction, and does not
        # communicate hidden reaction types through correlated generation.
        rng = np.random.RandomState(cfg.seed)
        for _ in range(4):
            y, x = rng.randint(1, cfg.size - 1, 2)
            if max(abs(y - c), abs(x - c)) > 2:
                self.walls[y, x] = True

        def rot(point):
            y, x = point
            for _ in range(cfg.rotate % 4):
                y, x = cfg.size - 1 - x, y
            return int(y), int(x)

        self.body, self.occupant = rot(self.body), rot(self.occupant)
        self.walls = np.rot90(self.walls, cfg.rotate).copy()
        self.goals = np.rot90(self.goals, cfg.rotate).copy()
        self.hazards = np.rot90(self.hazards, cfg.rotate).copy()
        self.t = 0
        self.success = self.collision = False
        return self.observation()

    def observation(self):
        patch = np.zeros((5, 5, 5), dtype=np.float32)
        for iy, dy in enumerate(range(-2, 3)):
            for ix, dx in enumerate(range(-2, 3)):
                y, x = self.body[0] + dy, self.body[1] + dx
                if not (0 <= y < self.H and 0 <= x < self.W):
                    patch[0, iy, ix] = 1
                    continue
                patch[0, iy, ix] = self.walls[y, x]
                patch[1, iy, ix] = self.goals[y, x]
                patch[2, iy, ix] = self.hazards[y, x]
                patch[3, iy, ix] = (y, x) == self.body
                patch[4, iy, ix] = (y, x) == self.occupant
        return patch.reshape(-1)

    def step(self, action):
        if action not in range(5):
            raise ValueError("five primitive actions required")
        old = self.body
        dy, dx = MOTIONS[action]
        dest = (old[0] + dy, old[1] + dx)
        contact = action != 4 and dest == self.occupant
        if contact:
            if self.cfg.rule == "push":
                oy, ox = dy, dx
            elif self.cfg.rule == "left":
                oy, ox = -dx, dy
            elif self.cfg.rule == "right":
                oy, ox = dx, -dy
            else:
                oy, ox = 0, 0
            moved_object = (self.occupant[0] + oy, self.occupant[1] + ox)
            if not self.walls[moved_object] and moved_object != old:
                self.occupant = moved_object
        if not self.walls[dest] and dest != self.occupant:
            self.body = dest
        self.t += 1
        self.collision = self.hazards[self.body] > 0
        self.success = bool(self.goals[self.body] > 0 and not self.collision)
        reward = -0.01 + (-2.0 if self.collision else float(self.goals[self.body]))
        self.goals[self.body] = 0
        done = self.success or self.collision or self.t >= self.cfg.max_steps
        return (
            self.observation(),
            reward,
            done,
            {
                "displacement": tuple(np.subtract(self.body, old).tolist()),
                "success": self.success,
                "collision": bool(self.collision),
                "contact": contact,
                "bump": action != 4 and self.body == old,
            },
        )
