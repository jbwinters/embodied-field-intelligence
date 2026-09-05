"""Continuous demonstration of contact, replenished rewards, and changing reactions.

The environment changes its contact response on a fixed clock and puts a
fresh reward beneath the object after each collection. Neither schedule nor
global geometry is included in observations. Body and object never reset.
"""

from dataclasses import replace

from .interaction_world import InteractionWorld, InteractionWorldConfig

REACTIONS = ("push", "left", "right")


class ContactArena(InteractionWorld):
    def __init__(self, steps=180, arena="islands"):
        if steps < 3 or arena not in ("open", "islands", "ring"):
            raise ValueError("at least three steps and a supported arena required")
        super().__init__(InteractionWorldConfig(size=13, acquisition=True, max_steps=steps))
        self.arena = arena
        self.phase_length = steps // 3

    def rule_at(self, step):
        return REACTIONS[min(2, step // self.phase_length)]

    def reset(self):
        super().reset()
        self.walls.fill(False)
        self.walls[[0, -1], :] = True
        self.walls[:, [0, -1]] = True
        if self.arena == "islands":
            for y, x in ((3, 3), (3, 4), (4, 9), (5, 9), (8, 3), (9, 3), (9, 8), (9, 9)):
                self.walls[y, x] = True
        elif self.arena == "ring":
            self.walls[3:5, 5:8] = True
        self.goals[self.occupant] = 1
        self.collections = 0
        return self.observation()

    def step(self, action):
        self.cfg = replace(self.cfg, rule=self.rule_at(self.t))
        _, reward, _, info = super().step(action)
        if info["success"]:
            self.collections += 1
            self.goals[self.occupant] = 1
        return self.observation(), reward, self.t >= self.cfg.max_steps, info
