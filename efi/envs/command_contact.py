"""A bounded contact task with command effects independent of body movement.

This environment supplies a lateral contact actuator. Its unknown wiring is
aligned or reversed; both lateral commands anchor the body. The agent receives
only the usual observation and body displacement, never the wiring or flags.
"""

from dataclasses import dataclass

import numpy as np

from ..core.anticipation import MOTIONS
from .interaction_world import InteractionWorld


@dataclass(frozen=True)
class CommandContactConfig:
    seed: int = 0
    rule: str = "aligned"
    rotate: int = 0
    size: int = 9
    context: int = 0
    source: bool = False
    layout: str = "left_blocked"
    front_wall: bool = False
    max_steps: int = 2

    def __post_init__(self):
        if self.rule not in ("aligned", "reversed"):
            raise ValueError("unknown actuator response")
        if self.size not in (9, 11, 13) or self.max_steps not in (1, 2):
            raise ValueError("bounded room and one/two-step episode required")
        if self.layout not in ("left_blocked", "right_blocked"):
            raise ValueError("unknown command-contact layout")
        if not isinstance(self.context, int) or not 0 <= self.context < 16 or self.context & 2:
            raise ValueError("source wall context must leave the body cell clear")


class CommandContactWorld(InteractionWorld):
    """Reuse the old observation interface without changing the old world's laws."""

    def __init__(self, cfg=None):
        super().__init__(cfg or CommandContactConfig())

    def reset(self):
        cfg = self.cfg
        self.H = self.W = cfg.size
        self.walls = np.zeros((self.H, self.W), dtype=bool)
        self.walls[[0, -1], :] = True
        self.walls[:, [0, -1]] = True
        self.goals = np.zeros((self.H, self.W), dtype=np.float32)
        self.hazards = np.zeros_like(self.goals)
        c = cfg.size // 2
        self.body, self.occupant = (c, c), (c - 1, c)
        code = cfg.context if cfg.source else (4 if cfg.layout == "left_blocked" else 8)
        if not cfg.source:
            code |= int(cfg.front_wall)
            self.goals[self.occupant] = 1
        for d, (dy, dx) in enumerate(MOTIONS[:4]):
            if code & (1 << d):
                self.walls[c - 1 + dy, c + dx] = True
        rng = np.random.RandomState(cfg.seed)
        for _ in range(4):
            y, x = rng.randint(1, cfg.size - 1, 2)
            if max(abs(y - c), abs(x - c)) > 2:
                self.walls[y, x] = True

        def rotate(point):
            y, x = point
            for _ in range(cfg.rotate % 4):
                y, x = cfg.size - 1 - x, y
            return int(y), int(x)

        self.body, self.occupant = rotate(self.body), rotate(self.occupant)
        self.walls = np.rot90(self.walls, cfg.rotate).copy()
        self.goals = np.rot90(self.goals, cfg.rotate).copy()
        self.hazards = np.rot90(self.hazards, cfg.rotate).copy()
        self.t = 0
        self.success = self.collision = False
        return self.observation()

    def step(self, action):
        if action not in range(5):
            raise ValueError("five directional commands required")
        before_body, before_object = self.body, self.occupant
        dy, dx = MOTIONS[action]
        toward = tuple(np.subtract(self.occupant, self.body))
        adjacent = toward in MOTIONS[:4]
        lateral = adjacent and (dy, dx) in ((-toward[1], toward[0]), (toward[1], -toward[0]))
        dest = (self.body[0] + dy, self.body[1] + dx)
        contact = lateral or (action != 4 and dest == self.occupant)
        if lateral:
            sign = 1 if self.cfg.rule == "aligned" else -1
            nxt = (self.occupant[0] + sign * dy, self.occupant[1] + sign * dx)
            if not self.walls[nxt] and nxt != self.body:
                self.occupant = nxt
            # Lateral actuation anchors the body even when the object moves.
            bump = self.occupant == before_object
        else:
            if not self.walls[dest] and dest != self.occupant:
                self.body = dest
            bump = action != 4 and self.body == before_body
        self.t += 1
        self.collision = bool(self.hazards[self.body] > 0)
        self.success = bool(self.goals[self.body] > 0 and not self.collision)
        reward = -0.01 + (-2.0 if self.collision else float(self.goals[self.body]))
        self.goals[self.body] = 0
        return (
            self.observation(),
            reward,
            self.success or self.collision or self.t >= self.cfg.max_steps,
            {
                "displacement": tuple(np.subtract(self.body, before_body).tolist()),
                "object_displacement": tuple(np.subtract(self.occupant, before_object).tolist()),
                "success": self.success,
                "collision": self.collision,
                "contact": bool(contact),
                "lateral": bool(lateral),
                "bump": bool(bump),
            },
        )
