"""Egocentric field control with learned, time-indexed hazard costs.

Input is a local five-channel window: walls, A, B, body, moving hazard.
No world dimensions, coordinates, hazard direction, phase, or rules are
available to this controller. Physical hazard costs and the radius-1 speed
bound are task priors; the motion transition rule is learned online.
"""

import numpy as np

from ..configs.anticipation_config import AnticipationConfig
from ..core.anticipation import action_probabilities, arrival_values
from .egocentric_controller import EgocentricFieldController
from .motion_schema import MotionSchema


class AnticipatoryFieldController(EgocentricFieldController):
    def __init__(self, cfg, ablate, anticipation=None, win=5, seed=0):
        super().__init__(cfg, ablate, win=win, seed=seed)
        self.anticipation = anticipation or AnticipationConfig()
        self.motion = MotionSchema(
            (self.M, self.M), self.anticipation.retention, self.anticipation.prior
        )
        self.forecasts = []
        self.policy = None

    def reset(self):
        super().reset()
        self.motion.reset()
        self.forecasts = []
        self.policy = None

    def observe(self, obs_vec):
        patch = np.asarray(obs_vec).reshape(5, self.win, self.win)
        super().observe(patch[:4].reshape(-1))
        occupied = np.zeros((self.M, self.M), dtype=bool)
        visible = np.zeros_like(occupied)
        half = self.win // 2
        py, px = self.pose
        for iy in range(self.win):
            for ix in range(self.win):
                y, x = py + iy - half, px + ix - half
                if 0 <= y < self.M and 0 <= x < self.M:
                    visible[y, x] = True
                    occupied[y, x] = patch[4, iy, ix] > 0.5
        self.motion.observe(
            occupied,
            visible,
            self.known_walls,
            learn=(
                self.anticipation.learn_motion and self.anticipation.forecast_mode != "unlearned"
            ),
        )

    def think(self, affect_state=None):
        terminal = super().think(affect_state)
        ac = self.anticipation
        # All ablations execute the same forecast and planning budget.
        self.forecasts = self.motion.forecast(self.known_walls, ac.horizon)
        edges = self.motion.edge_risks
        if ac.forecast_mode == "static":
            now = np.clip(self.motion.mass.sum(axis=0), 0, 1)
            self.forecasts = [now.copy() for _ in range(ac.horizon)]
            edges = None  # A static hazard has no motion flux.
        self.action_values = arrival_values(
            terminal,
            self.last_q,
            self.forecasts,
            self.last_walls_used,
            self.lam_current,
            ac.hazard_cost,
            edge_risks=edges,
        )
        py, px = self.pose
        self.policy = action_probabilities(self.action_values[:, py, px], self.lam_current)
        return terminal

    def select_action(self):
        if self.policy is None:
            raise RuntimeError("observe and think before selecting an action")
        return int(self.rng.choice(5, p=self.policy))

    def after_env_step(self, action, moved, picked):
        if action != 4:
            super().after_env_step(action, moved, picked)
            return
        # Waiting is an intentional zero displacement, not a failed motor
        # command; it must not trigger the parent's pose-recovery heuristic.
        if self.pschema is not None and self._prev_patch is not None:
            self._pending_transition = (self._prev_patch, 4, False)
        if picked in self.L:
            self.L[picked][self.pose] = self.belief_cfg.l_min
