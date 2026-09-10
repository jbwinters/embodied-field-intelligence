"""Continuous locally observed actuator world; hidden response recurrence."""

from dataclasses import dataclass

import numpy as np

from ..core.anticipation import MOTIONS
from ..agents.interaction_schema import heading, UNROTATE
from .interaction_world import InteractionWorld

# Canonical object effects for left/right commands. Supplied world family;
# these response maps are never passed to a controller.
RESPONSES = ((2, 3), (3, 2), (0, 0), (2, 2), (3, 3), (0, 2), (3, 0), (4, 0))


@dataclass(frozen=True)
class ContextWorldConfig:
    seed: int = 51000
    delay: int = 0
    conditions: int = 3
    stress: bool = False
    min_segment: int = 160
    max_segment: int = 224
    size: int = 13

    def __post_init__(self):
        if not 2 <= self.conditions <= 8 or self.delay not in (0, 3):
            raise ValueError("bounded response family and declared delay required")
        if self.size != 13 or self.min_segment < 16 or self.max_segment < self.min_segment:
            raise ValueError("fixed room and valid stream duration required")
        if not self.stress and self.conditions != 3:
            raise ValueError("primary A-B-A-C-B experiment needs three conditions")


class ContextWorld(InteractionWorld):
    def __init__(self, cfg=None):
        self.cfg = cfg or ContextWorldConfig()
        rng = np.random.RandomState(self.cfg.seed)
        pool = rng.permutation(self.cfg.conditions)
        if self.cfg.stress:
            returns = rng.permutation(self.cfg.conditions).tolist()
            if returns[0] == self.cfg.conditions - 1:
                returns = returns[1:] + returns[:1]
            ids = list(range(self.cfg.conditions)) + returns
        else:
            ids = [0, 1, 0, 2, 1]
        self.order = [int(pool[i]) for i in ids]
        self.lengths = rng.randint(
            self.cfg.min_segment, self.cfg.max_segment + 1, len(ids)
        ).tolist()
        self.ends = np.cumsum(self.lengths)
        self.max_steps = int(self.ends[-1])

    def reset(self):
        self.H = self.W = self.cfg.size
        self.walls = np.zeros((self.H, self.W), dtype=bool)
        self.walls[[0, -1]] = True
        self.walls[:, [0, -1]] = True
        for point in ((3, 3), (3, 9), (9, 3), (9, 9), (6, 3), (6, 9)):
            self.walls[point] = True
        self.goals = np.zeros((self.H, self.W), dtype=np.float32)
        self.hazards = np.zeros_like(self.goals)
        self.body, self.occupant = (6, 6), (5, 6)
        self.goals[self.occupant] = 1
        self.t = self.collections = 0
        self.phase = 0
        self.neutral_contacts = 0
        self.success = self.collision = False
        return self.observation()

    def step(self, action):
        if action not in range(5) or self.t >= self.max_steps:
            raise ValueError("valid primitive command within stream required")
        phase = int(np.searchsorted(self.ends, self.t, side="right"))
        if phase != self.phase:
            self.neutral_contacts = self.cfg.delay
            self.phase = phase
        before_body, before_object = self.body, self.occupant
        facing = heading(self.body, self.occupant)
        dy, dx = MOTIONS[action]
        intended = (self.body[0] + dy, self.body[1] + dx)
        lateral = facing is not None and action in UNROTATE[facing, [2, 3]]
        contact = lateral or (action != 4 and intended == self.occupant)
        neutral = bool(contact and self.neutral_contacts > 0)
        if lateral:
            canonical = int(np.flatnonzero(UNROTATE[facing] == action)[0])
            effect = RESPONSES[self.order[phase]][canonical - 2]
            if neutral:
                effect = 4
            ey, ex = MOTIONS[int(UNROTATE[facing, effect])]
            dest = (self.occupant[0] + ey, self.occupant[1] + ex)
            if not self.walls[dest] and dest != self.body:
                self.occupant = dest
            bump = self.occupant == before_object
        else:
            if not self.walls[intended] and intended != self.occupant:
                self.body = intended
            bump = action != 4 and self.body == before_body
        if neutral:
            self.neutral_contacts -= 1
        self.success = bool(self.goals[self.body] > 0)
        reward = -0.01 + float(self.success)
        if self.success:
            self.collections += 1
            self.goals[self.body] = 0
            self.goals[self.occupant] = 1
        self.t += 1
        return (
            self.observation(),
            reward,
            self.t >= self.max_steps,
            {
                "displacement": tuple(np.subtract(self.body, before_body).tolist()),
                "object_displacement": tuple(np.subtract(self.occupant, before_object).tolist()),
                "contact": bool(contact),
                "lateral": bool(lateral),
                "bump": bool(bump),
                "success": self.success,
                "collision": False,
                "phase": phase,
                "condition": self.order[phase],
                "neutral": neutral,
                "returning": self.order[phase] in self.order[:phase],
            },
        )
