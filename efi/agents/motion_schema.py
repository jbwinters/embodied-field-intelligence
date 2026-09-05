"""Online local motion-rule learning from observed occupancy transitions.

Five velocity channels carry probability mass. A categorical rule predicts
outgoing displacement from incoming displacement and four adjacent wall
bits. Rules start uniform; neither momentum nor reversal is supplied.
Counts decay when their context is revisited, allowing rule changes.

Association uses two radius-1 passes and excludes ambiguous visible matches.
It assumes isolated hazards: identity is not resolved through occlusion or
crowds. Both endpoints must be observed. One forecast step moves mass
at most one cell. Learned parameters are shared across cells, like the
existing PredictiveSchema; parameter sharing is distinct from the spatial
light cone of a forecast (it is not a strictly local plasticity substrate).
"""

import numpy as np

from ..core.anticipation import MOTIONS, shift


class MotionSchema:
    def __init__(self, shape, retention=0.9, prior=0.02):
        if not 0 < retention <= 1 or prior <= 0:
            raise ValueError("retention must be in (0, 1]; prior must be positive")
        self.shape = tuple(shape)
        self.retention = retention
        self.prior = prior
        self.counts = np.zeros((5, 16, 5), dtype=np.float64)
        self.transitions = 0
        self.scored = 0
        self.log_loss = 0.0
        self.last_loss = None
        self.reset()

    def reset(self):
        """Clear spatial memory; learned rules persist across episodes."""
        self.mass = np.zeros((5, *self.shape), dtype=np.float32)
        self.previous = None
        self.previous_walls = None
        self.velocity = np.full(self.shape, -1, dtype=np.int8)
        self.last_loss = None

    @staticmethod
    def wall_codes(walls):
        code = np.zeros(walls.shape, dtype=np.uint8)
        for d, (dy, dx) in enumerate(MOTIONS[:4]):
            code |= shift(walls, -dy, -dx, True).astype(np.uint8) << d
        return code

    def kernels(self, walls):
        code = self.wall_codes(walls)
        weights = self.weights(code)
        for d, (dy, dx) in enumerate(MOTIONS):
            weights[..., d] *= shift(~walls, -dy, -dx, False)[None, :, :]
        weights /= np.maximum(weights.sum(axis=-1, keepdims=True), 1e-30)
        return weights

    def weights(self, codes):
        """Unnormalized transition evidence; subclasses may pool contexts."""
        return self.counts[:, codes, :] + self.prior

    @staticmethod
    def advance(mass, kernels):
        result = np.zeros_like(mass)
        for d, (dy, dx) in enumerate(MOTIONS):
            outgoing = (mass * kernels[..., d]).sum(axis=0)
            result[d] = shift(outgoing, dy, dx)
        return result

    def observe(self, occupied, visible, walls, learn=True):
        """Correct from local sensing only; score predictions before learning.

        Arrays use the controller's internal coordinates. Outside ``visible``
        occupancy is ignored. A move is learned only when both its endpoints have
        a unique radius-1 association; no object identity or true velocity enters.
        """
        occupied = np.asarray(occupied, dtype=bool) & visible
        kernels = self.kernels(walls)
        predicted = self.advance(self.mass, kernels)
        velocity = np.full(self.shape, -1, dtype=np.int8)
        losses = []
        if self.previous is not None:
            successors = sum(shift(occupied.astype(np.int8), -dy, -dx) for dy, dx in MOTIONS)
            predecessors = sum(shift(self.previous.astype(np.int8), dy, dx) for dy, dx in MOTIONS)
            codes = self.wall_codes(self.previous_walls)
            for y, x in np.argwhere(self.previous & (successors == 1)):
                for d, (dy, dx) in enumerate(MOTIONS):
                    yy, xx = y + dy, x + dx
                    if not (0 <= yy < self.shape[0] and 0 <= xx < self.shape[1]):
                        continue
                    if not occupied[yy, xx] or predecessors[yy, xx] != 1:
                        continue
                    velocity[yy, xx] = d
                    incoming = int(self.velocity[y, x])
                    if incoming >= 0:
                        code = codes[y, x]
                        # Prequential score: the current transition has not
                        # yet contributed to these counts.
                        row = self.weights(code)[incoming]
                        legal = np.array([not (code & (1 << j)) for j in range(4)] + [True])
                        row = row * legal
                        probability = row[d] / row.sum()
                        losses.append(-np.log(max(probability, 1e-12)))
                        if learn:
                            self.counts[incoming, code] *= self.retention
                            self.counts[incoming, code, d] += 1
                            self.transitions += 1
        predicted[:, visible] = 0
        for y, x in np.argwhere(occupied):
            d = velocity[y, x]
            if d >= 0:
                predicted[d, y, x] = 1
            else:
                # An untracked sighting has unknown incoming velocity.
                predicted[:, y, x] = 0.2
        predicted[:, walls] = 0
        self.mass = predicted
        self.previous = occupied.copy()
        self.previous_walls = walls.copy()
        self.velocity = velocity
        self.last_loss = float(np.mean(losses)) if losses else None
        self.scored += len(losses)
        self.log_loss += sum(losses)

    def forecast(self, walls, horizon):
        """h radius-1 passes; return h arrival occupancy probability fields."""
        if horizon < 1:
            raise ValueError("horizon must be positive")
        kernels = self.kernels(walls)
        mass = self.mass.copy()
        fields = []
        self.edge_risks = []
        for _ in range(horizon):
            mass = self.advance(mass, kernels)
            fields.append(np.clip(mass.sum(axis=0), 0, 1))
            edges = np.zeros_like(mass)
            for d, reverse in enumerate((1, 0, 3, 2)):
                edges[d] = mass[reverse]
            self.edge_risks.append(edges)
        return fields
