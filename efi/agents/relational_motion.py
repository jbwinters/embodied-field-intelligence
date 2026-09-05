"""Rotation-equivariant motion evidence with a learned context backoff.

Heading supplies a local coordinate frame, not a preferred motion. A
relative turn learned in a corridor can predict motion in an open room.
Exact wall-context evidence overrides the backoff as observations accrue.
All five outgoing actions start equally likely. Geometry and rotation
invariance are supplied inductive biases; continuation/reflection are not.
"""

import numpy as np

from .motion_schema import MotionSchema
from ..core.anticipation import MOTIONS, shift


class RelationalMotionSchema(MotionSchema):
    # Cardinal headings in clockwise order, using the public action indices.
    _clockwise = (0, 3, 1, 2)
    correction_sweeps = 4

    def __init__(self, shape, retention=0.9, prior=0.02, generalize=True, correct_tracks=True):
        super().__init__(shape, retention, prior)
        self.generalize = generalize
        self.correct_tracks = correct_tracks
        self._context = np.zeros((4, 16), dtype=np.intp)
        self._outgoing = np.zeros((4, 5), dtype=np.intp)
        for heading, incoming in enumerate(self._clockwise):
            for d in range(4):
                relative = (self._clockwise.index(d) - heading) % 4
                self._outgoing[incoming, d] = self._clockwise[relative]
            self._outgoing[incoming, 4] = 4
            for code in range(16):
                self._context[incoming, code] = sum(
                    1 << self._outgoing[incoming, d] for d in range(4) if code & (1 << d)
                )

    def reset(self):
        super().reset()
        self.tick = 0
        self.evidence_time = np.full(self.shape, -1, dtype=np.int64)
        self.trace_time = np.full(self.shape, -1, dtype=np.int64)
        self.observed_space = None

    def kernels(self, walls):
        if self.observed_space is None:
            return super().kernels(walls)
        # An unseen neighbor is not evidence of free space. Marginalize
        # compatible wall contexts with a symmetric prior on unknown bits.
        # The 16 alternatives are local geometry hypotheses, not world maps.
        codes = np.arange(16, dtype=np.uint8)
        table = self.weights(codes)
        legal = np.stack(
            [~(codes & (1 << d)).astype(bool) for d in range(4)] + [np.ones(16, dtype=bool)],
            axis=-1,
        )
        table *= legal[None, :, :]
        table /= table.sum(axis=-1, keepdims=True)
        observed_bits = self.wall_codes(self.observed_space)
        actual_bits = self.wall_codes(walls)
        compatible = (codes[:, None, None] & observed_bits) == actual_bits
        mixture = compatible / np.maximum(compatible.sum(axis=0), 1)
        return np.einsum("icd,cyx->iyxd", table, mixture)

    @staticmethod
    def spread_stamp(stamp, walls):
        """One radius-1 max stencil; no global latest-sighting broadcast."""
        result = np.maximum.reduce([shift(stamp, dy, dx, -1) for dy, dx in MOTIONS])
        result[walls] = -1
        return result

    def observe(self, occupied, visible, walls, learn=True):
        """One isolated object per channel; newer sightings inhibit old tracks.

        Trace timestamps travel with a radius-1 support envelope. A sensory
        correction wave propagates four cells per call. Where newer evidence
        overtakes an older trace, its mass is suppressed. Hypotheses outside
        that correction cone persist until the evidence reaches them.
        No hidden position, identity tag, or global normalization is used.
        """
        occupied = np.asarray(occupied, dtype=bool) & visible
        if np.count_nonzero(occupied) > 1:
            raise ValueError("relational tracking requires one isolated object per channel")
        self.tick += 1
        if self.observed_space is None:
            self.observed_space = np.zeros(self.shape, dtype=bool)
        self.observed_space |= visible
        trace = self.spread_stamp(self.trace_time, walls)
        super().observe(occupied, visible, walls, learn=learn)
        trace[occupied] = self.tick
        self.evidence_time[occupied] = self.tick
        for _ in range(self.correction_sweeps):
            self.evidence_time = self.spread_stamp(self.evidence_time, walls)
        if self.correct_tracks:
            self.mass[:, trace < self.evidence_time] = 0
        trace[~self.mass.any(axis=0)] = -1
        self.trace_time = trace

    def weights(self, codes):
        if not self.generalize:
            return super().weights(codes)
        # Pool the finite parameter table, never spatial occupancy. Like the
        # original schema, these learned parameters are shared across cells.
        relative = np.zeros((16, 5), dtype=np.float64)
        for incoming in range(4):
            relative[self._context[incoming, :, None], self._outgoing[incoming]] += self.counts[
                incoming
            ]
        turns = relative.sum(axis=0) + self.prior
        backoff = turns / turns.sum()
        # One observation's worth of learned prior. Context-specific evidence
        # grows to ~10 observations at the default forgetting rate.
        relative += backoff[None, :]
        table = np.empty_like(self.counts)
        for incoming in range(4):
            table[incoming] = relative[self._context[incoming, :, None], self._outgoing[incoming]]
        # A stationary object has no heading. Average over the four possible
        # frames so that its wall-conditioned evidence is rotation invariant.
        table[4] = self.prior
        for incoming in range(4):
            table[4] += (
                self.counts[4][self._context[incoming, :, None], self._outgoing[incoming]] / 4
            )
        return table[:, codes, :].copy()
