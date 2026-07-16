"""Baseline agents: floors and ceilings for ForageWorld.

No EFI claim is defensible without these on the identical environment
distribution (planning/NEXT_STEPS.md 0.1):

- RandomAgent          -- the floor
- GreedyVisibleAgent   -- the "do you even need fields" strawman: no memory,
                          acts on the current window only
- AStarOracle          -- full-observability ceiling. Policy: BFS (== A*
                          with unit costs) to the NEAREST remaining A,
                          replanned every step; treats B cells as walls when
                          reward_B < 0. This nearest-target greedy tour is
                          an approximation of the optimal TSP ceiling --
                          a valid reference ceiling, not the true optimum.
- TabularQ             -- the "just use RL" comparison, trained on the same
                          distribution then frozen; EFI needs zero training
                          episodes, which this baseline makes explicit.

Normalized score convention: (X - random) / (astar - random).
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right


def _bfs_next_step(walls_like: np.ndarray, sy: int, sx: int,
                   targets: np.ndarray) -> Optional[int]:
    """First action of a BFS shortest path from (sy,sx) to the nearest
    target cell; None if no target is reachable."""
    H, W = walls_like.shape
    if targets[sy, sx]:
        return None
    prev = np.full((H, W), -1, dtype=np.int32)  # action taken to REACH cell
    seen = np.zeros((H, W), dtype=bool)
    seen[sy, sx] = True
    q = deque([(sy, sx)])
    goal = None
    while q:
        y, x = q.popleft()
        for a, (dy, dx) in enumerate(DIRS):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and not walls_like[ny, nx]:
                seen[ny, nx] = True
                prev[ny, nx] = a
                if targets[ny, nx]:
                    goal = (ny, nx)
                    q.clear()
                    break
                q.append((ny, nx))
        if goal:
            break
    if goal is None:
        return None
    # Walk back to the start to find the first action
    y, x = goal
    while True:
        a = int(prev[y, x])
        dy, dx = DIRS[a]
        py, px = y - dy, x - dx
        if (py, px) == (sy, sx):
            return a
        y, x = py, px


class RandomAgent:
    """Uniform over the 4 moves; bumping allowed. The honest floor."""

    trains = False
    needs_env = False

    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)

    def reset(self):
        pass

    def act(self, obs: np.ndarray, env=None) -> int:
        return int(self.rng.randint(4))

    def observe(self, obs, a, r, obs2, done):
        pass


class GreedyVisibleAgent:
    """Move toward the nearest visible A (Manhattan), never onto a visible
    wall, avoid visible B if an alternative exists; random otherwise.
    NO memory whatsoever -- this is the 'no fields needed' strawman that
    EFI's stigmergic memory must beat."""

    trains = False
    needs_env = False

    def __init__(self, seed: int = 0, win: int = 5):
        self.rng = np.random.RandomState(seed)
        self.win = win

    def reset(self):
        pass

    def act(self, obs: np.ndarray, env=None) -> int:
        win = self.win
        half = win // 2
        patch = obs[:4 * win * win].reshape(4, win, win)
        walls = patch[0] > 0.5
        A = patch[1] > 0.5
        B = patch[2] > 0.5

        def dest_ok(a, avoid_b=True):
            dy, dx = DIRS[a]
            py, px = half + dy, half + dx
            if walls[py, px]:
                return False
            if avoid_b and B[py, px]:
                return False
            return True

        targets = np.argwhere(A)
        if len(targets):
            # Nearest visible A by Manhattan distance from patch center
            dists = np.abs(targets[:, 0] - half) + np.abs(targets[:, 1] - half)
            ty, tx = targets[int(np.argmin(dists))]
            # Rank actions by distance reduction; tie-break by DIRS order
            scored = []
            for a, (dy, dx) in enumerate(DIRS):
                nd = abs(ty - (half + dy)) + abs(tx - (half + dx))
                scored.append((nd, a))
            scored.sort()
            for _, a in scored:
                if dest_ok(a, avoid_b=True):
                    return a
            for _, a in scored:
                if dest_ok(a, avoid_b=False):
                    return a
            return int(self.rng.randint(4))

        # No A visible: random over non-wall, non-B moves
        options = [a for a in range(4) if dest_ok(a, avoid_b=True)]
        if not options:
            options = [a for a in range(4) if dest_ok(a, avoid_b=False)]
        if not options:
            options = list(range(4))
        return int(options[self.rng.randint(len(options))])

    def observe(self, obs, a, r, obs2, done):
        pass


class AStarOracle:
    """Full-observability ceiling; the ONLY agent allowed to read env truth.

    Replans every step: BFS to the nearest remaining A, treating B as walls
    when reward_B < 0 (falls back to allowing B transit if that blocks all
    A). After all A are collected it ping-pongs between two free cells
    (cheapest way to run out the clock without bumping)."""

    trains = False
    needs_env = True

    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)
        self._last = None

    def reset(self):
        self._last = None

    def act(self, obs: np.ndarray, env=None) -> int:
        walls = env.walls
        y, x = env.y, env.x
        # Dynamic rewards (the swap event can flip signs mid-episode):
        # chase whichever kinds currently pay, avoid the aversive ones.
        rA = float(getattr(env, "reward_A", env.cfg.reward_A))
        rB = float(getattr(env, "reward_B", env.cfg.reward_B))
        targets = np.zeros_like(env.TA)
        avoid = np.zeros_like(env.TA)
        if rA > 0:
            targets |= env.TA
        elif rA < 0:
            avoid |= env.TA
        if rB > 0:
            targets |= env.TB
        elif rB < 0:
            avoid |= env.TB
        if targets.any():
            a = _bfs_next_step(walls | avoid, y, x, targets)
            if a is None and avoid.any():
                a = _bfs_next_step(walls, y, x, targets)  # transit unavoidable
            if a is not None:
                self._last = (y, x)
                return a
        # Nothing left to chase: ping-pong with the previous cell if open,
        # else the first open neighbor (never bump on purpose).
        for a, (dy, dx) in enumerate(DIRS):
            ny, nx = y + dy, x + dx
            if 0 <= ny < env.H and 0 <= nx < env.W and not walls[ny, nx]:
                if self._last == (ny, nx):
                    self._last = (y, x)
                    return a
        for a, (dy, dx) in enumerate(DIRS):
            ny, nx = y + dy, x + dx
            if 0 <= ny < env.H and 0 <= nx < env.W and not walls[ny, nx]:
                self._last = (y, x)
                return a
        return 0

    def observe(self, obs, a, r, obs2, done):
        pass


class TabularQ:
    """Tabular Q-learning on the raw observation window.

    state = obs bytes; epsilon-greedy linearly annealed over training;
    frozen (epsilon = eps_final, no updates) for evaluation."""

    trains = True
    needs_env = False

    def __init__(self, seed: int = 0, alpha: float = 0.1, gamma: float = 0.95,
                 eps_start: float = 0.1, eps_final: float = 0.01):
        self.rng = np.random.RandomState(seed)
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps = eps_start
        self.Q: Dict[bytes, np.ndarray] = {}
        self.learning = True

    def reset(self):
        pass

    def _q(self, obs: np.ndarray) -> np.ndarray:
        key = np.asarray(obs, dtype=np.float32).tobytes()
        if key not in self.Q:
            self.Q[key] = np.zeros(4, dtype=np.float64)
        return self.Q[key]

    def act(self, obs: np.ndarray, env=None) -> int:
        if self.rng.rand() < self.eps:
            return int(self.rng.randint(4))
        q = self._q(obs)
        best = np.flatnonzero(q == q.max())
        return int(best[self.rng.randint(len(best))])

    def observe(self, obs, a, r, obs2, done):
        if not self.learning:
            return
        q = self._q(obs)
        target = r if done else r + self.gamma * self._q(obs2).max()
        q[a] += self.alpha * (target - q[a])

    def set_training_progress(self, frac: float):
        self.eps = self.eps_start + (self.eps_final - self.eps_start) * min(1.0, frac)

    def freeze(self):
        self.learning = False
        self.eps = self.eps_final


def make_baseline(name: str, seed: int = 0, win: int = 5):
    return {
        "random": lambda: RandomAgent(seed),
        "greedy": lambda: GreedyVisibleAgent(seed, win=win),
        "astar": lambda: AStarOracle(seed),
        "q": lambda: TabularQ(seed),
    }[name]()


def run_baseline_episode(env, agent) -> dict:
    """Minimal episode loop for baseline agents; returns summary metrics."""
    obs = env.reset()
    agent.reset()
    ep_ret, steps = 0.0, 0
    collected = {"A": 0, "B": 0}
    nA_total = int(env.TA.sum())
    for _ in range(env.max_steps):
        a = agent.act(obs, env=env if agent.needs_env else None)
        obs2, r, done, info = env.step(a)
        agent.observe(obs, a, r, obs2, done)
        obs = obs2
        ep_ret += r
        steps += 1
        if info.get("picked") in ("A", "B"):
            collected[info["picked"]] += 1
        if done:
            break
    return {
        "return": float(ep_ret),
        "steps": steps,
        "targets_A": collected["A"],
        "targets_B": collected["B"],
        "success": collected["A"] >= nA_total and nA_total > 0,
    }


def train_tabular_q(env_factory, agent: TabularQ, episodes: int,
                    curve_window: int = 100) -> List[float]:
    """Train in place on freshly seeded envs; returns the training curve
    (mean return per `curve_window`-episode window)."""
    returns = []
    for ep in range(episodes):
        agent.set_training_progress(ep / max(1, episodes - 1))
        env = env_factory(ep)
        returns.append(run_baseline_episode(env, agent)["return"])
    curve = [float(np.mean(returns[i:i + curve_window]))
             for i in range(0, len(returns), curve_window)]
    agent.freeze()
    return curve
