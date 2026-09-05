"""Learn command-conditioned joint body/object effects from real contact.

Geometry and motion are expressed in the frame facing the adjacent object.
16 local wall contexts x 5 commands x 25 joint effects: 2,000 counts.
No push, yield, reflection, or object-response rule is supplied here.
"""

import numpy as np

from ..core.anticipation import MOTIONS

# Canonical directions: forward, backward, left, right, stay. Their indices
# are the same as the public up/down/left/right/wait motor convention.
DIRECTIONS = np.asarray(MOTIONS, dtype=np.int16)
ROTATE = np.empty((4, 5), dtype=np.intp)
for _h, (_dy, _dx) in enumerate(DIRECTIONS[:4]):
    _right = np.asarray((_dx, -_dy))
    for _d, _v in enumerate(DIRECTIONS):
        _canonical = (-int(_v @ np.asarray((_dy, _dx))), int(_v @ _right))
        ROTATE[_h, _d] = MOTIONS.index(_canonical)
UNROTATE = np.argsort(ROTATE, axis=1)
EFFECT_BODY = np.repeat(np.arange(5), 5)
EFFECT_OBJECT = np.tile(np.arange(5), 5)


def heading(body, occupant):
    delta = tuple(np.asarray(occupant) - np.asarray(body))
    return MOTIONS.index(delta) if delta in MOTIONS[:4] else None


def context_at(port, occupant, facing):
    """Radius-1 geometry, read from already gathered body-local evidence."""
    code = 0
    for d, (dy, dx) in enumerate(MOTIONS[:4]):
        y, x = np.asarray(occupant) + (dy, dx)
        if not (0 <= y < port.shape[0] and 0 <= x < port.shape[1]):
            return None
        if port[y, x, 0] < 0:
            return None
        code |= int(port[y, x, 0] > 0.5) << int(ROTATE[facing, d])
    return code


class InteractionSchema:
    def __init__(self, prior=0.01, retention=0.95, conditioned=True):
        self.prior = prior
        self.retention = retention
        self.conditioned = conditioned
        self.counts = np.zeros((16, 5, 25), dtype=np.float32)
        self.version = 0
        self.observed = 0
        self.partial = 0

    def table(self):
        counts = self.counts
        if not self.conditioned:
            counts = np.broadcast_to(counts.mean(axis=1, keepdims=True), counts.shape)
        result = counts + self.prior
        return result / result.sum(axis=-1, keepdims=True)

    def update(self, experience, displacement, occupant_after, learn=True):
        """Score saved predictions first; missing object feedback adds no counts."""
        delta = tuple(displacement)
        if delta not in MOTIONS:
            raise ValueError("feedback exceeds the supplied one-cell motor support")
        bd = int(ROTATE[experience.heading, MOTIONS.index(delta)])
        probs = np.asarray(experience.probabilities).reshape(5, 5)
        if occupant_after is None:
            self.partial += 1
            return -float(np.log(max(float(probs[bd].sum()), 1e-12))), False
        odelta = tuple(np.asarray(occupant_after) - np.asarray(experience.occupant))
        if odelta not in MOTIONS:
            self.partial += 1
            return None, False
        od = int(ROTATE[experience.heading, MOTIONS.index(odelta)])
        effect = 5 * bd + od
        loss = -float(np.log(max(float(probs[bd, od]), 1e-12)))
        if learn:
            row = self.counts[experience.context, experience.action]
            row *= self.retention
            row[effect] += 1
            self.observed += 1
            self.version += 1
        return loss, True
