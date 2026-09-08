"""Bounded response-model memory inferred from real, local feedback.

Archives are correlated snapshots, not independent observations. Their soft
responsibilities describe present applicability; no condition-chain model,
world phase, or reward is used to recognize a returning response.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from .interaction_controller import InteractionFieldController
from .interaction_schema import InteractionSchema, ROTATE
from ..core.anticipation import MOTIONS


@dataclass(frozen=True)
class ContextMemoryConfig:
    capacity: int = 3
    fast_retention: float = 0.1
    slow_rate: float = 0.02
    hazard: float = 0.2
    fast_prior: float = 0.5
    validation_window: int = 8
    validation_loss: float = 0.35
    separation: float = 0.12
    usage_retention: float = 0.995

    def __post_init__(self):
        if not 1 <= self.capacity <= 8 or not 2 <= self.validation_window <= 32:
            raise ValueError("bounded archive/history capacity required")
        for value in (
            self.fast_retention,
            self.slow_rate,
            self.hazard,
            self.fast_prior,
            self.usage_retention,
        ):
            if not 0 < value < 1:
                raise ValueError("rates must be strictly between zero and one")
        if self.validation_loss <= 0 or not 0 < self.separation < np.log(2):
            raise ValueError("positive validation loss and bounded JS separation required")


class ContextMemory(InteractionSchema):
    def __init__(self, cfg=None, prior=0.01, mode="bank"):
        self.cfg = cfg or ContextMemoryConfig()
        if mode not in ("bank", "no_reuse", "no_history"):
            raise ValueError("unknown memory control")
        super().__init__(prior, self.cfg.fast_retention)
        self.mode = mode
        self.archives = np.zeros((self.cfg.capacity, 16, 5, 25), dtype=np.float32)
        self.active = np.zeros(self.cfg.capacity, dtype=bool)
        self.usage = np.zeros(self.cfg.capacity)
        self.weights = np.zeros(self.cfg.capacity + 1)
        self.weights[0] = 1
        self.validation = deque(maxlen=self.cfg.validation_window)
        self.recent = deque(maxlen=32)
        self.saved = None
        self.admissions = self.evictions = 0
        self.last_event = None

    def components(self):
        tables = np.concatenate((self.counts[None], self.archives), axis=0) + self.prior
        return tables / tables.sum(axis=-1, keepdims=True)

    def prior_weights(self):
        if not self.active.any():
            return np.eye(1, self.cfg.capacity + 1, 0)[0]
        base = np.r_[
            self.cfg.fast_prior, self.active * (1 - self.cfg.fast_prior) / self.active.sum()
        ]
        if self.mode == "no_history":
            return base
        return (1 - self.cfg.hazard) * self.weights + self.cfg.hazard * base

    def table(self):
        components = self.components()
        if self.mode == "no_reuse":
            return components[0]
        return np.einsum("i,icaj->caj", self.prior_weights(), components).astype(np.float32)

    def capture(self, experience):
        """Bind expert predictions to the same pre-action local geometry mask."""
        if experience is None:
            self.saved = None
            return
        p = self.components()[:, experience.context, experience.action].astype(np.float64)
        p *= np.asarray(experience.probabilities)[None] > 0
        mass = p.sum(axis=1)
        p /= np.maximum(mass[:, None], 1e-30)
        weights = self.prior_weights() * mass
        weights /= max(weights.sum(), 1e-30)
        self.saved = (experience.sequence, p.copy(), weights.copy())

    def _admit(self):
        if len(self.validation) < self.cfg.validation_window:
            return
        if np.mean(self.validation) > self.cfg.validation_loss:
            return
        tables = self.components()
        rows = sorted(set(self.recent))
        if not rows:
            return
        codes, actions = np.asarray(rows).T
        for slot in np.flatnonzero(self.active):
            supported = (self.counts[codes, actions].sum(axis=-1) >= 1) & (
                self.archives[slot, codes, actions].sum(axis=-1) >= 1
            )
            # New geometry alone is not evidence for a different response model.
            if not supported.any():
                return
            p = tables[0, codes[supported], actions[supported]]
            q = tables[slot + 1, codes[supported], actions[supported]]
            mid = 0.5 * (p + q)
            js = 0.5 * np.sum(p * np.log(p / mid) + q * np.log(q / mid), axis=-1)
            if float(js.mean()) < self.cfg.separation:
                return
        free = np.flatnonzero(~self.active)
        slot = int(free[0]) if len(free) else int(np.argmin(self.usage))
        evicted = bool(self.active[slot])
        self.archives[slot] = self.counts
        self.active[slot] = True
        self.usage[slot] = 1
        self.weights[slot + 1] = max(self.weights[0], 0.1)
        self.weights /= self.weights.sum()
        self.admissions += 1
        self.evictions += int(evicted)
        self.last_event = {
            "observation": self.observed,
            "slot": slot,
            "evicted": evicted,
            "validated_loss": float(np.mean(self.validation)),
        }
        self.validation.clear()

    def update(self, experience, displacement, occupant_after, learn=True):
        if self.saved is None or self.saved[0] != experience.sequence:
            raise RuntimeError("capture immutable component predictions before real feedback")
        _, p, weights = self.saved
        self.saved = None
        bd = int(ROTATE[experience.heading, MOTIONS.index(tuple(displacement))])
        odelta = (
            None
            if occupant_after is None
            else tuple(np.subtract(occupant_after, experience.occupant))
        )
        complete = odelta in MOTIONS if odelta is not None else False
        if complete:
            od = int(ROTATE[experience.heading, MOTIONS.index(odelta)])
            effect = 5 * bd + od
            likelihood = p[:, effect]
        else:
            likelihood = p.reshape(-1, 5, 5)[:, bd].sum(axis=1)
        # The existing contract scores the ACTUAL constrained saved mixture
        # and updates fast counts only on complete associated physical effects.
        loss, actual_complete = super().update(experience, displacement, occupant_after, learn)
        self.last_event = None
        if not learn:
            return loss, actual_complete
        posterior = weights * likelihood
        if posterior.sum() > 0:
            self.weights = posterior / posterior.sum()
        if not actual_complete:
            # Belief can change; missing object feedback cannot add joint support.
            self.version += 1
            return loss, False
        self.validation.append(-float(np.log(max(float(likelihood[0]), 1e-12))))
        self.recent.append((experience.context, experience.action))
        self.usage *= self.cfg.usage_retention
        self.usage += self.weights[1:]
        for slot in np.flatnonzero(self.active):
            credit = float(self.weights[slot + 1] * likelihood[slot + 1])
            row = self.archives[slot, experience.context, experience.action]
            row *= 1 - self.cfg.slow_rate * credit
            row[effect] += credit
        self._admit()
        return loss, True

    @property
    def storage_bytes(self):
        return sum(
            a.nbytes for a in (self.counts, self.archives, self.active, self.usage, self.weights)
        )


class ContextFieldController(InteractionFieldController):
    """Publish a bounded mixture through the existing field/controller path."""

    def __init__(self, cfg=None, seed=0, memory=None, mode="bank", reference=False):
        super().__init__(cfg, seed, reference)
        self.schema = ContextMemory(memory, self.cfg.prior, mode)

    def select_action(self, forced=None):
        action = super().select_action(forced)
        self.schema.capture(self.pending)
        return action
