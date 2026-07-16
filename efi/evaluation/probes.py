"""Instrumented probes of internal dynamics.

Speed-of-thought probe: information moves through the internal fields at a
bounded speed (kappa value sweeps per tick, radius-1 stencil => at most
kappa cells per tick). Reveal a target at distance d for one tick and
measure how many THINK-ticks pass before the policy at the (stationary)
agent responds. Latency should scale like d / kappa -- the internal light
cone made behavioral. A cached or non-local shortcut would show up as
latency flat in d, so this doubles as a locality regression test.

This bypasses the environment window on purpose: it is a probe of internal
dynamics, not a gameplay path.
"""

from typing import Optional

import numpy as np

from ..configs import AgentConfig, Ablations, EnvConfig
from ..envs import ForageWorld
from ..agents import FieldController, ForageAdapter

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def policy_distribution(V: np.ndarray, y: int, x: int, walls: np.ndarray,
                        lam: float) -> np.ndarray:
    """Exact softmax over neighbor values (no sampling)."""
    scores = np.full(4, -np.inf)
    H, W = V.shape
    for i, (dy, dx) in enumerate(DIRS):
        ny, nx = y + dy, x + dx
        if 0 <= ny < H and 0 <= nx < W and not walls[ny, nx]:
            scores[i] = float(V[ny, nx]) / max(lam, 1e-6)
    m = scores.max()
    e = np.exp(scores - m)
    return e / e.sum()


def kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 1e-12
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], 1e-12))))


def reaction_latency(d: int, kappa: int, corridor_w: int = 60,
                     warmup: int = 20, max_ticks: int = 200,
                     kl_threshold: float = 0.1, seed: int = 0) -> Optional[int]:
    """
    THINK-ticks from a one-tick belief injection at distance d (to the
    agent's right, same row) until KL(policy || baseline) exceeds the
    threshold. None if the value never arrives (beyond the reach budget).
    The agent never moves: this isolates internal propagation.
    """
    H = 5
    env = ForageWorld(EnvConfig(H=H, W=corridor_w, p_wall=0.0,
                                n_targets_A=0, n_targets_B=0,
                                max_steps=10_000, seed=seed))
    env.reset()
    env.walls[:] = False
    env.y, env.x = H // 2, 2

    cfg = AgentConfig(valA_init=1.0, seed=seed, z_sweeps=kappa,
                      init_sweeps=1,          # no free orientation budget
                      epistemic_mode="none",  # no exploration pull
                      affect_enabled=False,
                      schema_mode="off")
    ablate = Ablations(trail=0, novelty=0, corner=0, schema=0)
    agent = FieldController(env, ForageAdapter(env), cfg, ablate, seed=seed)

    obs = env._obs()
    for _ in range(warmup):
        agent.step_fields(obs)
        agent.compose_value()

    walls = env.walls
    baseline = policy_distribution(agent.V, env.y, env.x, walls, agent.lam_current)

    # One-tick reveal: hard positive evidence at distance d (probe API)
    ty, tx = env.y, env.x + d
    agent.L["A"][ty, tx] = agent.belief_cfg.l_max
    agent.fields["A"][ty, tx] = 1.0

    for t in range(1, max_ticks + 1):
        # think only: no new observations, no movement
        agent.compose_value()
        pi = policy_distribution(agent.V, env.y, env.x, walls, agent.lam_current)
        if kl(pi, baseline) > kl_threshold:
            return t
    return None
