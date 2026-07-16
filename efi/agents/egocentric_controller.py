"""Egocentric field controller: the embodiment premise made literal.

The world-frame FieldController reads self.env.y / self.env.x (a free GPS)
and allocates fields on the true H x W grid (a priori knowledge of world
size). This controller removes both:

- Internal fields live on a fixed M x M map the agent brings with it;
  the world's size is unknown and never queried.
- Pose is tracked by DEAD RECKONING from an efference copy: the only
  movement feedback is info["moved"] (did my motor command succeed?),
  delivered via after_env_step(). Under odometry noise (env p_slip) two
  local mechanisms correct the pose: VISUAL ODOMETRY (the overlap of
  consecutive windows aligns with the true displacement -- slips are
  detected the tick they happen) and anchor-based template matching against
  the internal map (residual drift). Measured at p_slip=0.1: mean pose
  error 0.4 cells vs 5.5 uncorrected (~13x).
- World boundaries are DISCOVERED: the environment reports out-of-bounds
  as wall in the observation window, and the agent maps them like any wall.

Forbidden here: env.y, env.x, env.H, env.W, env.walls -- enforced by
tests/test_egocentric.py, which greps this file.

The runner drives it as a closed box:

    agent.observe(obs)                      # senses
    V = agent.think(affect_state)           # value sweeps on the inner map
    a = agent.select_action()               # softmax over neighbor V
    ... env.step(a) ...
    agent.after_env_step(a, moved, picked)  # efference copy + pickups
"""

from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np

from ..configs.belief_config import BeliefConfig
from ..core import corner_hazard, diffuse_masked, update_visit_trail
from ..core.affect import affect_to_lambda
from ..core.belief import sigmoid, logodds_correct, logodds_predict
from ..core.desirability import VBIG, pick_action_from_value, value_sweeps
from ..core.infogain import epistemic_beta, pooled_gain, uncertainty_map
from ..core.membrane import peripersonal_field

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class EgocentricFieldController:
    """FieldController semantics on an internal, pose-anchored map."""

    def __init__(self, cfg, ablate, win: int = 5, seed: int = 0):
        self.cfg = cfg
        self.ablate = ablate
        self.win = int(win)
        self.rng = np.random.RandomState(seed)

        self.M = int(getattr(cfg, "map_size", 129))
        M = self.M
        self.belief_cfg = getattr(cfg, "belief", None) or BeliefConfig()

        self.known_walls = np.zeros((M, M), dtype=bool)
        self.seen = np.zeros((M, M), dtype=bool)
        self.L = {
            "A": np.full((M, M), self.belief_cfg.l_prior, dtype=np.float32),
            "B": np.full((M, M), self.belief_cfg.l_prior, dtype=np.float32),
        }
        self.trail = np.zeros((M, M), dtype=np.float32)
        self.V = np.zeros((M, M), dtype=np.float32)

        self.valence: Dict[str, float] = {
            "A": float(cfg.valA_init),
            "B": float(cfg.valB_init),
            "Novel": float(cfg.w_novel),
        }

        self.pose: Tuple[int, int] = (M // 2, M // 2)
        self.lam = float(getattr(cfg, "lam_base", 0.02))
        self.lam_current = self.lam
        self.z_sweeps = int(getattr(cfg, "z_sweeps", 3))
        self.last_residuals = []
        self._fresh_value = True
        self._pos_hist = deque(maxlen=3)
        self._prev_patch = None
        self.last_surprise = 0.0
        self.affect_state = None
        self._search_boost_ticks = 0

        # Predictive schema (learned local world rule); rules persist across
        # episodes -- the world's physics doesn't reset.
        self.pschema = None
        if str(getattr(cfg, "schema_mode", "predictive")) == "predictive":
            from .predictive_schema import PredictiveSchema
            self.pschema = PredictiveSchema(win=self.win)
        self._pending_transition = None

    # ------------------------------------------------------------------
    def reset(self):
        M = self.M
        self.known_walls[:] = False
        self.seen[:] = False
        for k in self.L:
            self.L[k][:] = self.belief_cfg.l_prior
        self.trail[:] = 0.0
        self.V[:] = 0.0
        self.pose = (M // 2, M // 2)
        self.lam_current = self.lam
        self.last_residuals = []
        self._fresh_value = True
        self._pos_hist.clear()
        self._pos_hist.append(self.pose)
        self._prev_patch = None
        self.last_surprise = 0.0
        self._search_boost_ticks = 0
        self._pending_transition = None

    # ------------------------------------------------------------------
    def _patch(self, obs_vec: np.ndarray) -> np.ndarray:
        win = self.win
        ch = int(len(obs_vec) // (win * win))
        return obs_vec[:ch * win * win].reshape(ch, win, win)

    def _estimate_displacement(self, prev_patch: np.ndarray, patch: np.ndarray,
                               d_assumed) -> Optional[Tuple[int, int]]:
        """Egomotion from vision: score each candidate displacement by the
        agreement of the wall/A/B texture in the overlap of consecutive
        windows (world cell at patch pos p now was at p + d one tick ago).
        Informative cells only (+2 matching structure, -1 mismatch, 0 for
        empty-empty). Returns a displacement only when it beats the
        efference copy's score by a clear margin; texture-free overlaps
        return None (fall back to dead reckoning)."""
        win = self.win

        def score(d):
            dy, dx = d
            # The window travels with the agent: the world cell at current
            # patch position p sat at p + d in the previous window.
            # Overlap in current-window coords requires p + d in-bounds:
            y_lo, y_hi = max(0, -dy), min(win, win - dy)
            x_lo, x_hi = max(0, -dx), min(win, win - dx)
            if y_lo >= y_hi or x_lo >= x_hi:
                return -10**6
            cur = patch[:3, y_lo:y_hi, x_lo:x_hi] > 0.5
            prv = prev_patch[:3, y_lo + dy:y_hi + dy, x_lo + dx:x_hi + dx] > 0.5
            informative = cur.any(axis=0) | prv.any(axis=0)
            match = informative & (cur == prv).all(axis=0)
            mismatch = (cur != prv).any(axis=0)
            return 2 * int(match.sum()) - int(mismatch.sum())

        s_assumed = score(d_assumed)
        best_d, best_s = None, s_assumed
        for d in DIRS:
            if d == d_assumed:
                continue
            s = score(d)
            if s > best_s:
                best_d, best_s = d, s
        if best_d is not None and best_s >= s_assumed + 3:
            return best_d
        return None

    def _correct_pose(self, patch: np.ndarray):
        """Loop closure by local template matching: test +/-1 shifts of the
        observed wall pattern against the internal map.

        Two rules make this sound:
        - Score only INFORMATIVE cells: a matching wall is an anchor (+2),
          a mismatch is evidence against the alignment (-1), matching empty
          space is 0 -- uniform open terrain must not vote.
        - Score every offset over the SAME cell set: the intersection of
          cells seen under all candidate offsets. Otherwise a shift that
          drags extra seen wall cells into the window wins on count rather
          than on alignment (e.g. sliding along a straight boundary wall).
        Snap only on a clear win (>= +2)."""
        win = self.win
        half = win // 2
        walls_local = patch[0] > 0.5
        py, px = self.pose
        # Radius 1 normally (wider searches false-positive on sparse
        # anchors); radius 2 for a few ticks after a bump contradicted the
        # map -- direct evidence the pose has drifted more than one cell.
        r = 2 if self._search_boost_ticks > 0 else 1
        if self._search_boost_ticks > 0:
            self._search_boost_ticks -= 1
        span = range(-r, r + 1)
        offsets = [(oy, ox) for oy in span for ox in span]

        y0, x0 = py - half, px - half
        # All candidate windows must lie inside the internal map
        if not (r <= y0 and y0 + win + r <= self.M and r <= x0 and x0 + win + r <= self.M):
            return

        views = {off: self.known_walls[y0 + off[0]: y0 + off[0] + win,
                                       x0 + off[1]: x0 + off[1] + win]
                 for off in offsets}
        seen_views = {off: self.seen[y0 + off[0]: y0 + off[0] + win,
                                     x0 + off[1]: x0 + off[1] + win]
                      for off in offsets}
        common = np.ones((win, win), dtype=bool)
        for off in offsets:
            common &= seen_views[off]
        if not common.any():
            return

        def score(off):
            kw = views[off]
            match_wall = common & walls_local & kw
            mismatch = common & (walls_local != kw)
            return 2 * int(match_wall.sum()) - int(mismatch.sum())

        base = score((0, 0))
        best, best_off = base, (0, 0)
        for off in offsets:
            if off == (0, 0):
                continue
            s = score(off)
            if s > best:
                best, best_off = s, off
        if best >= base + 2:
            self.pose = (py + best_off[0], px + best_off[1])

    def observe(self, obs_vec: np.ndarray):
        """Sense: visual odometry, pose correction, mapping, belief
        correction+prediction, trail deposit, surprise scalar."""
        patch = self._patch(obs_vec)
        correction_on = bool(getattr(self.cfg, "pose_correction", True))

        # 1) Visual odometry FIRST: the overlap between consecutive windows
        # aligns with the TRUE displacement. If it confidently disagrees
        # with the efference copy (an odometry slip), re-anchor the pose
        # now -- instant, map-free slip detection.
        d_actual = None
        if self._pending_transition is not None:
            pp, a_prev, mv_prev = self._pending_transition
            if mv_prev:
                d_actual = DIRS[a_prev]
                if correction_on:
                    d_hat = self._estimate_displacement(pp, patch, d_actual)
                    if d_hat is not None and d_hat != d_actual:
                        py, px = self.pose
                        self.pose = (py - d_actual[0] + d_hat[0],
                                     px - d_actual[1] + d_hat[1])
                        d_actual = d_hat

        # 2) Anchor-based correction for residual/accumulated drift
        if correction_on:
            self._correct_pose(patch)

        # 3) Close out the pending transition: (prev window, action, moved)
        # -> this window, with the CORRECTED displacement so slip ticks do
        # not teach garbage rules. Prediction error IS the surprise signal.
        if self.pschema is not None and self._pending_transition is not None:
            pp, a, mv = self._pending_transition
            self.last_surprise = self.pschema.observe_transition(
                pp, a, mv, patch, displacement=d_actual)
        self._pending_transition = None

        win = self.win
        half = win // 2
        py, px = self.pose
        bc = self.belief_cfg
        walls_local = patch[0] > 0.5
        A_local = patch[1] > 0.5
        B_local = patch[2] > 0.5

        pos_cells = {"A": [], "B": []}
        neg_cells = {"A": [], "B": []}
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                gy, gx = py + dy, px + dx
                if not (0 <= gy < self.M and 0 <= gx < self.M):
                    continue
                self.seen[gy, gx] = True
                if walls_local[dy + half, dx + half]:
                    self.known_walls[gy, gx] = True
                    continue
                (pos_cells["A"] if A_local[dy + half, dx + half]
                 else neg_cells["A"]).append((gy, gx))
                (pos_cells["B"] if B_local[dy + half, dx + half]
                 else neg_cells["B"]).append((gy, gx))

        # Belief blur gate from learned static-confidence
        static_conf = self.pschema.static_confidence if self.pschema else 0.0
        gate = 1.0 - float(static_conf)
        for channel in ("A", "B"):
            Lc = logodds_correct(self.L[channel], pos_cells[channel], neg_cells[channel],
                                 l_pos=bc.l_pos, l_neg=bc.l_neg,
                                 l_min=bc.l_min, l_max=bc.l_max)
            self.L[channel] = logodds_predict(Lc, self.known_walls,
                                              diff=bc.belief_diff * gate,
                                              decay=bc.belief_decay,
                                              l_prior=bc.l_prior,
                                              rho_prior=bc.rho_prior * gate)

        # Trail (repulsor) with ping-pong boost
        self._pos_hist.append(self.pose)
        pingpong = (len(self._pos_hist) == 3 and self._pos_hist[0] == self._pos_hist[2]
                    and self._pos_hist[0] != self._pos_hist[1])
        if self.ablate.trail:
            v_inj = self.cfg.v_inj * (2.5 if pingpong else 1.0)
            self.trail = update_visit_trail(self.trail, py, px, self.known_walls,
                                            v_decay=self.cfg.v_decay,
                                            v_diff=self.cfg.v_diff, v_inj=v_inj)

        # Surprise scalar (feeds affect; no novelty FIELD in this controller).
        # With the predictive schema active, surprise was already set from
        # prediction error above; this is the schemaless fallback.
        if self.pschema is None and self._prev_patch is not None:
            self.last_surprise = float(np.mean(np.abs(patch[:3] - self._prev_patch[:3])))
        self._prev_patch = patch.copy()

    # ------------------------------------------------------------------
    def think(self, affect_state=None) -> np.ndarray:
        """Assemble q / R_inj on the internal map and run value sweeps."""
        cfg = self.cfg
        self.affect_state = affect_state
        M = self.M
        py, px = self.pose

        if affect_state is not None:
            lam = affect_to_lambda(
                affect_state, lam_base=self.lam,
                k_pain=float(getattr(cfg, "k_pain_lambda", 0.9)),
                k_arousal=float(getattr(cfg, "k_arousal_lambda", 0.3)),
                lam_min=float(getattr(cfg, "lam_min", 0.005)),
                lam_max=float(getattr(cfg, "lam_max", 0.1)))
        else:
            lam = self.lam
        self.lam_current = lam

        q = np.full((M, M), float(getattr(cfg, "q_step", 0.01)), dtype=np.float32)
        q += float(getattr(cfg, "q_trail", 0.08)) * self.trail
        if self.ablate.corner:
            q += float(getattr(cfg, "q_corner", 0.02)) * corner_hazard(self.known_walls)

        walls = self.known_walls
        if getattr(cfg, "membrane_enabled", False) and affect_state is not None:
            membrane = peripersonal_field(
                self.known_walls, self.seen, py, px,
                cfg.membrane_r_min, affect_state.arousal, affect_state.pain,
                cfg.membrane_r_gain_arousal, cfg.membrane_r_gain_pain)
            q += float(getattr(cfg, "q_membrane", 0.3)) * membrane
            forbidden = membrane >= float(getattr(cfg, "barrier_threshold", 0.75))
            if forbidden.any():
                walls = walls | forbidden

        R = np.zeros((M, M), dtype=np.float32)
        p = {c: sigmoid(self.L[c]) for c in ("A", "B")}
        for channel in ("A", "B"):
            val = float(self.valence.get(channel, 0.0))
            if val >= 0.0:
                R += val * p[channel]
            else:
                q += (-val) * p[channel]

        beta = epistemic_beta(
            float(getattr(cfg, "beta_epist", 0.3)),
            arousal=affect_state.arousal if affect_state is not None else 0.0,
            pain=affect_state.pain if affect_state is not None else 0.0,
            k_curiosity=float(getattr(cfg, "k_curiosity", 0.5)),
            k_fear=float(getattr(cfg, "k_fear", 0.8)))
        if beta > 0.0 and str(getattr(cfg, "epistemic_mode", "infogain")) != "none":
            u = uncertainty_map(self.L["A"], self.L["B"], self.seen, self.known_walls,
                                w_map=float(getattr(cfg, "w_map_uncertainty", 1.0)))
            R += beta * pooled_gain(u, self.win)

        R_inj = np.where(R > 1e-6, R, -VBIG).astype(np.float32)

        sweeps = self.z_sweeps
        if self._fresh_value:
            # Orientation budget: the agent doesn't know the world's size,
            # so it budgets by its own map -- but the map starts empty, so a
            # window-scale budget suffices until structure accumulates.
            sweeps += int(getattr(cfg, "init_sweeps", 0)) or (4 * self.win)
            self._fresh_value = False

        self.V, self.last_residuals = value_sweeps(self.V, q, R_inj, walls,
                                                   lam=lam, sweeps=sweeps)
        self.last_q, self.last_R_inj, self.last_walls_used = q, R_inj, walls
        return self.V

    def select_action(self) -> int:
        """Softmax over neighbor V on the agent's OWN map: only KNOWN walls
        are masked -- bumping into undiscovered walls is possible and is how
        they get discovered (via the next observation)."""
        py, px = self.pose
        return pick_action_from_value(self.V, py, px, self.known_walls,
                                      lam=self.lam_current, rng=self.rng)

    # ------------------------------------------------------------------
    def after_env_step(self, action: int, moved: bool, picked: Optional[str]):
        """Efference copy: dead-reckon the pose; register pickups locally."""
        if self.pschema is not None and self._prev_patch is not None:
            self._pending_transition = (self._prev_patch, int(action), bool(moved))
        if moved:
            dy, dx = DIRS[int(action)]
            py, px = self.pose
            self.pose = (py + dy, px + dx)
        else:
            dy, dx = DIRS[int(action)]
            by, bx = self.pose[0] + dy, self.pose[1] + dx
            if 0 <= by < self.M and 0 <= bx < self.M:
                if self.seen[by, bx] and not self.known_walls[by, bx]:
                    # Contradiction: the map says this cell is free, yet the
                    # move failed. That is evidence of pose drift, not of a
                    # wall -- writing a phantom wall here would poison the
                    # map. Widen the re-localization search instead.
                    self._search_boost_ticks = 3
                else:
                    # Unmapped obstacle: a failed move reveals a wall even
                    # before it has been seen through the window.
                    self.known_walls[by, bx] = True
        if picked in self.L:
            py, px = self.pose
            self.L[picked][py, px] = self.belief_cfg.l_min

    def learn_valence(self, channel: str, reward: float):
        lr = float(self.cfg.valence_lr)
        clip = float(getattr(self.cfg, "valence_clip", 1.5))
        cur = self.valence.get(channel, 0.0)
        self.valence[channel] = float(np.clip(cur + lr * reward, -clip, clip))
