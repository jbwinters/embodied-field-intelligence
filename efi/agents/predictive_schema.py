"""Predictive schema: learn the world's local rule online, from experience.

The Oja/BCM schema had no defined objective and measurably hurt returns
(ablation: -0.11). This one has exactly one job: PREDICT the next
observation. Because ForageWorld is a deterministic gridworld, a
count-based local-rule learner suffices -- the agent literally learns the
environment's cellular-automaton rule, online, with no gradients:

    T[(3x3x4-channel neighborhood, action)] -> counts over next cell state

What it buys:
- surprise = its prediction error (replacing the hand-crafted pred_err),
  feeding affect;
- static_confidence: once the agent has LEARNED the world doesn't change,
  the belief prediction step stops blurring (diffusion gate) -- memory
  sharpens exactly when the world is learned to be static;
- imagination: rolling the learned rule forward with the sensors detached.

Held-out scoring: every 10th transition is excluded from training and used
as a test item, so prediction accuracy is measured on data the learner has
not memorized.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N_STATES = 16  # 4 binary channels (wall, A, B, agent) packed to an int


def _cell_state(patch: np.ndarray, y: int, x: int) -> int:
    s = 0
    for c in range(4):
        if patch[c, y, x] > 0.5:
            s |= 1 << c
    return s


def _nbhd_key(patch: np.ndarray, y: int, x: int) -> bytes:
    """3x3 x 4-channel binary neighborhood packed to bytes (36 bits)."""
    bits = (patch[:, y - 1:y + 2, x - 1:x + 2] > 0.5)
    return np.packbits(bits.reshape(-1)).tobytes()


class PredictiveSchema:
    """Count-based local transition model over observation windows."""

    def __init__(self, win: int = 5, holdout_every: int = 10,
                 confidence_min: float = 0.8):
        self.win = int(win)
        self.holdout_every = int(holdout_every)
        self.confidence_min = float(confidence_min)
        self.T: Dict[Tuple[bytes, int], np.ndarray] = {}
        self._updates = 0
        # Held-out scoring
        self.heldout_total = 0
        self.heldout_correct = 0
        # Static-world evidence: of confident predictions, how often is
        # next state == current state?
        self._static_hits = 0
        self._static_total = 0
        self.last_surprise = 0.0

    # ------------------------------------------------------------------
    def _predict_state(self, key: Tuple[bytes, int]) -> Tuple[Optional[int], float]:
        counts = self.T.get(key)
        if counts is None:
            return None, 0.0
        total = counts.sum()
        if total == 0:
            return None, 0.0
        s = int(np.argmax(counts))
        return s, float(counts[s] / total)

    def observe_transition(self, patch_t: np.ndarray, action: int, moved: bool,
                           patch_t1: np.ndarray, displacement=None):
        """
        Learn from one consecutive window pair. Window alignment: the window
        travels with the agent, so the world cell at patch position p at
        time t sits at p - d at time t+1, where d is the ACTUAL displacement
        (defaults to d(action) when moved; pass `displacement` explicitly
        under odometry slip so misaligned pairs don't teach garbage rules).
        """
        win = self.win
        if displacement is not None:
            dy, dx = displacement
        else:
            dy, dx = DIRS[int(action)] if moved else (0, 0)

        errors = []
        for y in range(1, win - 1):
            for x in range(1, win - 1):
                y1, x1 = y - dy, x - dx
                if not (0 <= y1 < win and 0 <= x1 < win):
                    continue
                key = (_nbhd_key(patch_t, y, x), int(action))
                actual = _cell_state(patch_t1, y1, x1)

                pred, conf = self._predict_state(key)
                # Surprise semantics: a confidently-predicted cell scores
                # its error; an UNFAMILIAR cell scores 1.0 -- the unknown is
                # maximally surprising (this is what makes novel places
                # arousing before any rule exists, and familiar places calm).
                if pred is not None and conf >= self.confidence_min:
                    errors.append(0.0 if pred == actual else 1.0)
                else:
                    errors.append(1.0)

                self._updates += 1
                if self._updates % self.holdout_every == 0:
                    # Held-out item: score only (if predictable), never train
                    if pred is not None and conf >= self.confidence_min:
                        self.heldout_total += 1
                        self.heldout_correct += int(pred == actual)
                    continue

                counts = self.T.get(key)
                if counts is None:
                    counts = np.zeros(N_STATES, dtype=np.int64)
                    self.T[key] = counts
                counts[actual] += 1

                if pred is not None and conf >= self.confidence_min:
                    current = _cell_state(patch_t, y, x)
                    # World-stasis check on wall/A/B bits only: the agent
                    # channel changes whenever the agent moves, and the
                    # agent is not the world.
                    self._static_total += 1
                    self._static_hits += int((actual & 0b0111) == (current & 0b0111))

        self.last_surprise = float(np.mean(errors)) if errors else 0.0
        return self.last_surprise

    # ------------------------------------------------------------------
    def predict_patch(self, patch: np.ndarray, action: int, moved: bool = True):
        """
        Predict the next window (same traveling frame): returns
        (predicted 4 x win x win patch, per-cell confidence win x win).
        Cells without a confident rule (or entering the window from outside)
        get confidence 0.
        """
        win = self.win
        dy, dx = DIRS[int(action)] if moved else (0, 0)
        pred = np.zeros((4, win, win), dtype=np.float32)
        conf = np.zeros((win, win), dtype=np.float32)
        for y in range(1, win - 1):
            for x in range(1, win - 1):
                y1, x1 = y - dy, x - dx
                if not (0 <= y1 < win and 0 <= x1 < win):
                    continue
                key = (_nbhd_key(patch, y, x), int(action))
                s, c = self._predict_state(key)
                if s is None:
                    continue
                for ch in range(4):
                    pred[ch, y1, x1] = float((s >> ch) & 1)
                conf[y1, x1] = c
        return pred, conf

    def imagine(self, patch: np.ndarray, actions: List[int]):
        """Sensor-detached rollout: apply the learned rule along an action
        sequence. Returns [(patch, confidence), ...] per imagined tick;
        confidence collapses where the rule is unknown (honest ignorance)."""
        out = []
        cur = patch.copy()
        for a in actions:
            cur, conf = self.predict_patch(cur, a, moved=True)
            out.append((cur.copy(), conf.copy()))
        return out

    # ------------------------------------------------------------------
    @property
    def static_confidence(self) -> float:
        """Fraction of confident predictions where the world did NOT change.
        Drives the belief-diffusion gate: a world learned to be static
        should not blur in memory."""
        if self._static_total < 50:
            return 0.0
        return self._static_hits / self._static_total

    @property
    def heldout_accuracy(self) -> float:
        if self.heldout_total == 0:
            return 0.0
        return self.heldout_correct / self.heldout_total

    @property
    def n_rules(self) -> int:
        return len(self.T)
