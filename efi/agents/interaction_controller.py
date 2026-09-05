"""Embodied contact learning through bounded evidence and joint-effect fields.

This opt-in pilot has a reliable displacement sensor, one visible isolated
object, no global environment access, and zero terminal continuation.
It does not inherit the older unbounded predictive dictionary.
"""

from collections import deque

import numpy as np

from ..configs.interaction_config import InteractionConfig
from ..core.anticipation import MOTIONS
from ..core.experience import Experience, RuleField, gather
from ..core.interaction import InteractionField, probabilities
from .interaction_schema import InteractionSchema, ROTATE, context_at, heading


class InteractionFieldController:
    def __init__(self, cfg=None, seed=0, reference=False):
        self.cfg = cfg or InteractionConfig()
        self.rng = np.random.RandomState(seed)
        self.schema = InteractionSchema(
            self.cfg.prior, self.cfg.retention, self.cfg.action_conditioned
        )
        self.rules = RuleField(self.cfg.map_size)
        self.memory = np.zeros((self.cfg.map_size, self.cfg.map_size, 5), dtype=np.float32)
        self.reference = reference
        self.history = deque(maxlen=32)
        self.sequence = 0
        self.reset()

    def reset(self):
        self.pose = (self.cfg.map_size // 2, self.cfg.map_size // 2)
        self.memory.fill(0)
        self.memory[:, :, 0] = -1
        self.rules.reset()
        self.pending = None
        self.displacement = None
        self.last_loss = None
        self.last_complete = False
        self.policy = None
        self.occupant = None
        self.work = {}

    def observe(self, observation):
        patch = np.asarray(observation, dtype=np.float32).reshape(5, 5, 5)
        if not np.isfinite(patch).all() or np.any(patch[1] > self.cfg.goal_reward_bound):
            raise ValueError("observation exceeds the finite task reward contract")
        self.policy = None
        sensory = np.zeros_like(self.memory)
        sensory[:, :, 0] = -1
        py, px = self.pose
        if min(py, px) < 2 or max(py, px) >= self.cfg.map_size - 2:
            raise ValueError("pilot exceeded fixed-map sensory capacity")
        for y in range(5):
            for x in range(5):
                sensory[py + y - 2, px + x - 2] = patch[:, y, x]
                self.memory[py + y - 2, px + x - 2] = patch[:, y, x]
        sensed, work = gather(sensory, self.pose, 2, -1)
        occupied = np.argwhere(sensed[:, :, 4] > 0.5)
        if len(occupied) > 1:
            raise ValueError("contact pilot requires one isolated visible object")
        self.occupant = tuple(occupied[0] + np.asarray(self.pose) - 2) if len(occupied) else None
        self.last_loss = None
        self.last_complete = False
        if self.pending is not None:
            if self.displacement is None:
                raise RuntimeError("movement feedback must precede the next observation")
            self.last_loss, self.last_complete = self.schema.update(
                self.pending, self.displacement, self.occupant, self.cfg.learn
            )
            self.history.append(
                (self.pending, self.displacement, self.last_loss, self.last_complete)
            )
        self.pending = None
        self.displacement = None
        self.port, extra = gather(self.memory, self.pose, 4, -1)
        self.work = {
            "gather_elements": work + extra,
            "evidence_radius": 2,
            "working_port_radius": 4,
            "rule_passes": self.cfg.rule_passes,
        }
        self.rules.publish(self.pose, self.schema.table(), self.schema.version)
        before = self.rules.bytes_copied
        self.rules.spread(self.cfg.rule_passes)
        self.work["rule_bytes_copied"] = self.rules.bytes_copied - before

    def think(self):
        self.field = InteractionField(self.port, self.rules, self.pose, self.cfg)
        if self.occupant is None:
            self.action_values = np.full(
                5, self.cfg.horizon * (self.cfg.step_cost + self.cfg.collision_cost)
            )
            upper = max(0.0, self.cfg.goal_reward_bound + self.cfg.step_cost)
            self.field.value_bounds = np.column_stack(
                (self.action_values, np.full(5, self.cfg.horizon * upper))
            )
        else:
            relative = np.asarray(self.occupant) - self.pose + 4
            self.action_values = self.field.solve(relative, self.reference)
        self.policy = probabilities(self.action_values, self.cfg.temperature)
        self.work["outcome_terms"] = self.field.outcome_terms
        self.work["feedback_groups"] = getattr(self.field, "groups", 0)
        return self.action_values

    def select_action(self, forced=None):
        if self.policy is None:
            raise RuntimeError("observe and think before acting")
        action = int(self.rng.choice(5, p=self.policy)) if forced is None else int(forced)
        if action not in range(5):
            raise ValueError("five primitive actions required")
        facing = heading(self.pose, self.occupant) if self.occupant is not None else None
        if facing is not None:
            relative = np.asarray(self.occupant) - self.pose + 4
            code = context_at(self.port, relative, facing)
            if code is not None:
                ac = int(ROTATE[facing, action])
                # Store the actual constrained pre-action predictive distribution.
                absolute = self.field.first[0][action]
                canonical = np.zeros(25)
                for effect in range(25):
                    idx = 5 * ROTATE[facing, effect // 5] + ROTATE[facing, effect % 5]
                    canonical[idx] = absolute[effect]
                self.pending = Experience(
                    self.sequence,
                    code,
                    ac,
                    facing,
                    tuple(canonical),
                    int(self.rules.versions[self.pose]),
                    self.pose,
                    self.occupant,
                )
                self.sequence += 1
        return action

    def after_env_step(self, displacement):
        displacement = tuple(displacement)
        if displacement not in MOTIONS:
            raise ValueError("reliable one-cell displacement feedback required")
        self.displacement = displacement
        self.pose = tuple(np.asarray(self.pose) + displacement)

    @property
    def nbytes(self):
        """Fixed arrays only; benchmark peak allocation separately, including scratch."""
        return self.rules.nbytes + self.memory.nbytes + self.schema.counts.nbytes + self.port.nbytes
