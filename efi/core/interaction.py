"""Two-step local joint-effect fields with observation-contingent control.

All inputs are a gathered 9x9 local port or radius-1 model-cache reads.
Hypotheses are bounded co-located channels, not access to hypothetical
environment instances. One physical transition moves each entity <=1 cell.
The terminal boundary is zero. No imagined observation trains the model.
"""

import numpy as np

from ..agents.interaction_schema import DIRECTIONS, EFFECT_BODY, EFFECT_OBJECT, ROTATE


def soft_value(scores, temperature):
    peak = np.max(scores, axis=-1, keepdims=True)
    return peak[..., 0] + temperature * np.log(np.exp((scores - peak) / temperature).mean(axis=-1))


def probabilities(scores, temperature):
    weights = np.exp((scores - np.max(scores)) / temperature)
    return weights / weights.sum()


def grouped_continuation(weights, keys, action_values, temperature):
    """Average hidden consequences BEFORE selecting a shared future action."""
    _, groups = np.unique(keys, axis=0, return_inverse=True)
    mass = np.bincount(groups, weights=weights)
    values = np.zeros((len(mass), 5), dtype=np.float64)
    np.add.at(values, groups, weights[:, None] * action_values)
    values /= np.maximum(mass[:, None], 1e-30)
    return soft_value(values, temperature)[groups], len(mass)


class InteractionField:
    def __init__(self, port, rules, center, cfg):
        self.port = port
        self.rules = rules
        self.center = np.asarray(center)
        self.cfg = cfg
        self.outcome_terms = 0

    def sample(self, coordinates, channel):
        y, x = coordinates[..., 0], coordinates[..., 1]
        valid = (y >= 0) & (x >= 0) & (y < 9) & (x < 9)
        value = self.port[np.clip(y, 0, 8), np.clip(x, 0, 8), channel]
        return np.where(valid, value, -1 if channel == 0 else 0)

    def transition(self, body, occupant, action):
        """Parallel pointwise updates over a finite set of joint hypotheses.

        Known rigid exclusion and motor support condition the prior distribution.
        Unknown geometry retains unresolved probability; it is never normalized
        away. The learned table supplies actual joint response probabilities.
        """
        body = np.asarray(body, dtype=np.int16).reshape(-1, 2)
        occupant = np.asarray(occupant, dtype=np.int16).reshape(-1, 2)
        action = np.asarray(action, dtype=np.intp).reshape(-1)
        n = len(action)
        bnext = body[:, None, :] + DIRECTIONS[EFFECT_BODY][None]
        onext = occupant[:, None, :] + DIRECTIONS[EFFECT_OBJECT][None]
        near = ((occupant - body)[:, None, :] == DIRECTIONS[None, :4]).all(axis=2)
        adjacent = near.any(axis=1)
        facing = near.argmax(axis=1)
        p = np.zeros((n, 25), dtype=np.float64)
        context_known = np.ones(n, dtype=bool)
        versions = np.full(n, -1, dtype=np.int32)
        # Outside the declared radius-1 interaction, the object is passive
        # and the body's free motor support is supplied. Contact laws are not.
        intended = body + DIRECTIONS[action]
        blocked = self.sample(intended, 0) > 0.5
        free_effect = 5 * np.where(blocked, 4, action) + 4
        p[np.arange(n), free_effect] = 1
        if adjacent.any():
            idx = np.flatnonzero(adjacent)
            h = facing[idx]
            neighbor = occupant[idx, None, :] + DIRECTIONS[None, :4, :]
            geometry = self.sample(neighbor, 0)
            context_known[idx] = (geometry >= 0).all(axis=1)
            bits = np.left_shift(1, ROTATE[h[:, None], np.arange(4)[None]])
            codes = ((geometry > 0.5) * bits).sum(axis=1).astype(np.intp)
            # In a depth-two rollout, a model query is at the real body or
            # one hypothetical body displacement away. This is an N4 read.
            offset = body[idx] - 4
            if np.any(np.abs(offset).sum(axis=1) > 1):
                raise ValueError("model query exceeds the declared radius-1 read")
            site = self.center + offset
            versions[idx] = self.rules.versions[site[:, 0], site[:, 1]]
            raw = self.rules.values[site[:, 0], site[:, 1], codes, ROTATE[h, action[idx]]].astype(
                np.float64
            )
            raw[versions[idx] < 0] = 1 / 25  # factory prior, never remote lookup
            permutation = 5 * ROTATE[h[:, None], EFFECT_BODY] + ROTATE[h[:, None], EFFECT_OBJECT]
            p[idx] = np.take_along_axis(raw, permutation, axis=1)
        # A motor can realize its commanded unit displacement or stay.
        motor = (EFFECT_BODY[None] == action[:, None]) | (EFFECT_BODY[None] == 4)
        bw, ow = self.sample(bnext, 0), self.sample(onext, 0)
        overlap = (bnext == onext).all(axis=2)
        swap = (bnext == occupant[:, None]).all(axis=2) & (onext == body[:, None]).all(axis=2)
        legal = motor & (bw <= 0.5) & (ow <= 0.5) & ~overlap & ~swap
        p *= legal
        denom = p.sum(axis=1, keepdims=True)
        p /= np.maximum(denom, 1e-30)
        known = (bw >= 0) & (ow >= 0) & context_known[:, None]
        p *= known
        unknown = np.clip(1 - p.sum(axis=1), 0, 1)
        reward = np.full((n, 25), self.cfg.step_cost, dtype=np.float64)
        collision = self.sample(bnext, 2) > 0
        goal = np.maximum(self.sample(bnext, 1), 0)
        reward += np.where(collision, self.cfg.collision_cost, goal)
        terminal = collision | (goal > 0)
        self.outcome_terms += n * 25
        return p, unknown, bnext, onext, reward, terminal, versions

    def solve(self, occupant, reference=False):
        body = np.tile((4, 4), (5, 1))
        occupied = np.tile(occupant, (5, 1))
        first = self.transition(body, occupied, np.arange(5))
        p, unknown, b1, o1, reward, terminal, _ = first
        rmin = self.cfg.step_cost + self.cfg.collision_cost
        # Unseen cells may contain a goal. An upper bound must use the supplied
        # task reward range, never the maximum of currently observed rewards.
        rmax = max(0.0, self.cfg.goal_reward_bound + self.cfg.step_cost)
        q = (p * reward).sum(axis=1) + unknown * rmin * self.cfg.horizon
        upper = (p * reward).sum(axis=1) + unknown * rmax * self.cfg.horizon
        self.first = first
        self.groups = 0
        if self.cfg.horizon == 1:
            self.value_bounds = np.column_stack((q, upper))
            return q
        bflat, oflat = b1.reshape(-1, 2), o1.reshape(-1, 2)
        second = self.transition(
            np.repeat(bflat, 5, axis=0),
            np.repeat(oflat, 5, axis=0),
            np.tile(np.arange(5), len(bflat)),
        )
        p2, u2, _, _, r2, _, _ = second
        q2 = ((p2 * r2).sum(axis=1) + u2 * rmin).reshape(-1, 5)
        q2_upper = ((p2 * r2).sum(axis=1) + u2 * rmax).reshape(-1, 5)
        # The object component is observable only within the actual next
        # 5x5 window. Hidden positions share one policy per observed history.
        visible = np.max(np.abs(oflat - bflat), axis=1) <= 2
        observed_object = np.where(visible[:, None], oflat, -1)
        keys = np.column_stack((np.repeat(np.arange(5), 25), bflat, observed_object))
        weights = (p * ~terminal).reshape(-1)
        if reference:
            # Independent scalar tabular reduction lives in evaluation code.
            from ..evaluation.interaction_reference import scalar_continuation

            continuation, self.groups = scalar_continuation(weights, keys, q2, self.cfg.temperature)
        else:
            continuation, self.groups = grouped_continuation(
                weights, keys, q2, self.cfg.temperature
            )
        upper_continuation, _ = grouped_continuation(weights, keys, q2_upper, self.cfg.temperature)
        q += (weights * continuation).reshape(5, 25).sum(axis=1)
        upper += (weights * upper_continuation).reshape(5, 25).sum(axis=1)
        self.value_bounds = np.column_stack((q, upper))
        self.second, self.keys, self.weights, self.q2 = second, keys, weights, q2
        return q

    def object_forecast(self, policy):
        """Actual planner forecast under its shared feedback-contingent policy.

        Read-only display data. Omitted unresolved mass stays omitted, never
        renormalized into certainty. A terminated branch keeps its final position.
        """
        p, _, _, o1, _, terminal, _ = self.first
        if not hasattr(self, "second"):
            return o1.reshape(-1, 2), (policy[:, None] * p).reshape(-1)
        _, groups = np.unique(self.keys, axis=0, return_inverse=True)
        mass = np.bincount(groups, weights=self.weights)
        values = np.zeros((len(mass), 5))
        np.add.at(values, groups, self.weights[:, None] * self.q2)
        values /= np.maximum(mass[:, None], 1e-30)
        exp = np.exp((values - values.max(axis=1, keepdims=True)) / self.cfg.temperature)
        pi2 = (exp / exp.sum(axis=1, keepdims=True))[groups]
        first_mass = self.weights * np.repeat(policy, 25)
        p2, _, _, o2, *_ = self.second
        future_mass = p2 * (first_mass[:, None] * pi2).reshape(-1, 1)
        stopped = (policy[:, None] * p * terminal).reshape(-1)
        return (
            np.concatenate((o1.reshape(-1, 2), o2.reshape(-1, 2))),
            np.concatenate((stopped, future_mass.reshape(-1))),
        )
