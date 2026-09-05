"""Bounded local transport and immutable empirical prediction records.

N8 transport has a Chebyshev light cone. Spatial axes are always first;
remaining axes are co-located payload channels, not distant read handles.
"""

from dataclasses import dataclass

import numpy as np

N8 = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))


def transport(array, dy, dx, fill=0):
    """One synchronous radius-1 N8 pass, without wraparound."""
    if max(abs(dy), abs(dx)) > 1:
        raise ValueError("transport is a single radius-1 pass")
    out = np.full_like(array, fill)
    h, w = array.shape[:2]
    out[max(0, dy) : min(h, h + dy), max(0, dx) : min(w, w + dx)] = array[
        max(0, -dy) : min(h, h - dy), max(0, -dx) : min(w, w - dx)
    ]
    return out


def gather(array, center, radius, fill=0):
    """Gather tagged offsets through <=radius N8 passes per dependency.

    Each offset follows its own shortest local route. The summed array work
    is larger than the light cone; callers meter both. No remote slice reads
    populate the returned body-local port.
    """
    shape = (2 * radius + 1, 2 * radius + 1) + array.shape[2:]
    out = np.empty(shape, dtype=array.dtype)
    work = 0
    for y, dy in enumerate(range(-radius, radius + 1)):
        for x, dx in enumerate(range(-radius, radius + 1)):
            cur = array
            ry, rx = dy, dx
            while ry or rx:
                sy, sx = int(np.sign(ry)), int(np.sign(rx))
                cur = transport(cur, -sy, -sx, fill)
                ry -= sy
                rx -= sx
                work += array.size
            out[y, x] = cur[tuple(center)]
    return out, work


@dataclass(frozen=True)
class Experience:
    sequence: int
    context: int
    action: int
    heading: int
    probabilities: tuple
    model_version: int
    body: tuple
    occupant: tuple


class RuleField:
    """One writer publishes immutable snapshots; caches never learn.

    The complete finite 16-context table is the payload for this first pilot.
    This trades copying cost for a simple auditable local version contract.
    Two preallocated buffers bound storage; no remote pointer reads a newer
    table. A sparse four-record cache is a later optimization, not claimed here.
    """

    def __init__(self, size):
        self.values = np.zeros((size, size, 16, 5, 25), dtype=np.float32)
        self.scratch = np.zeros_like(self.values)
        self.versions = np.full((size, size), -1, dtype=np.int32)
        self.bytes_copied = 0

    def reset(self):
        # Uninitialized payload is inaccessible until its version arrives.
        self.versions.fill(-1)
        self.bytes_copied = 0

    def publish(self, center, table, version):
        self.values[tuple(center)] = table
        self.versions[tuple(center)] = version
        self.bytes_copied += table.nbytes

    def spread(self, passes):
        h, w = self.versions.shape
        for _ in range(passes):
            old = self.versions
            new = old.copy()
            sources = np.arange(h * w).reshape(h, w)
            chosen = sources.copy()
            for dy, dx in N8:
                candidate = transport(old, dy, dx, -1)
                better = candidate > new
                incoming = transport(sources, dy, dx, -1)
                chosen[better] = incoming[better]
                new[better] = candidate[better]
            valid = new >= 0
            flat = self.values.reshape(h * w, 16, 5, 25)
            self.scratch[valid] = flat[chosen[valid]]
            self.bytes_copied += int(valid.sum()) * flat[0].nbytes
            self.values, self.scratch = self.scratch, self.values
            self.versions = new

    @property
    def nbytes(self):
        return self.values.nbytes + self.scratch.nbytes + self.versions.nbytes
