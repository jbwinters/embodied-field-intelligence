"""Metrics and results tracking."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

import numpy as np


def moving_average(x, window: int):
    """Trailing moving average; entries before `window` use the available prefix."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    c = np.cumsum(np.insert(x, 0, 0.0))
    for i in range(len(x)):
        lo = max(0, i + 1 - window)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def adaptation_lag(rewards, shift_times, window: int = 20,
                   baseline_window: int = 100, tolerance: float = 0.2):
    """
    Steps after each shift until the trailing `window`-step mean reward
    recovers to within `tolerance` of its pre-shift `baseline_window` mean.

    Returns a list aligned with shift_times; None where recovery never
    happens before the series ends (that IS the frozen-policy signature).
    """
    r = np.asarray(rewards, dtype=np.float64)
    ma = moving_average(r, window)
    lags = []
    for s in shift_times:
        if s <= 0 or s >= len(r):
            lags.append(None)
            continue
        base = float(np.mean(r[max(0, s - baseline_window):s]))
        threshold = base - tolerance * abs(base)
        # The trailing MA at the shift instant still contains pre-shift
        # rewards, so first find the DIP (performance actually depressed),
        # then measure recovery from the shift time. No dip => lag 0.
        dip = None
        for t in range(s, len(r)):
            if ma[t] < threshold:
                dip = t
                break
        if dip is None:
            lags.append(0)
            continue
        lag = None
        for t in range(dip, len(r)):
            if ma[t] >= threshold:
                lag = t - s
                break
        lags.append(lag)
    return lags


def regret_series(agent_rewards, oracle_rewards):
    """Cumulative regret r_t = cumsum(oracle) - cumsum(agent), truncated to
    the shorter series."""
    n = min(len(agent_rewards), len(oracle_rewards))
    a = np.cumsum(np.asarray(agent_rewards[:n], dtype=np.float64))
    o = np.cumsum(np.asarray(oracle_rewards[:n], dtype=np.float64))
    return o - a


def regret_slopes(regret, window: int = 100):
    """Mean per-step regret growth in consecutive windows."""
    regret = np.asarray(regret, dtype=np.float64)
    slopes = []
    for i in range(0, len(regret) - window, window):
        slopes.append(float((regret[i + window] - regret[i]) / window))
    return slopes


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    
    total_return: float
    steps: int
    targets_collected: Dict[str, int] = field(default_factory=dict)
    efficiency: float = 0.0
    seed: int = 0
    episode: int = 0
    mean_cosine: Optional[float] = None  # Alignment between gradient and motion
    valence_snapshot: Dict[str, float] = field(default_factory=dict)  # Current valence weights
    
    # Safety metrics
    bumps_per_100: float = 0.0  # Bumps per 100 steps
    mean_pain: float = 0.0  # Average pain level
    max_pain: float = 0.0  # Maximum pain level
    mean_wall_distance: float = 0.0  # Average distance to nearest wall
    affect_history: List[Dict[str, float]] = field(default_factory=list)  # Full affect state history
    
    # --- NEW capability/behavior metrics ---
    coverage: float = 0.0                # % of *unique visited* free cells
    frontier_efficiency: float = 0.0     # new cells discovered per step when novelty is high
    path_optimality: Optional[float] = None  # steps / oracle shortest route to visited pickup sequence
    backtrack_rate: float = 0.0          # fraction of moves that immediately reverse the previous move

    # --- Egocentric (dead reckoning) diagnostics ---
    mean_pose_error: float = 0.0         # mean L1 error of dead-reckoned displacement
    final_pose_error: float = 0.0        # L1 pose error at episode end

    # --- Predictive schema diagnostics ---
    accuracy_predictive: float = 0.0     # held-out next-cell prediction accuracy
    schema_rules: int = 0                # distinct (neighborhood, action) rules learned

    # --- LMDP control diagnostics ---
    barrier_deadlocks: int = 0           # ticks where every open neighbor was barrier-forbidden
    mean_residual: float = 0.0           # mean per-tick final value-sweep residual (fixed-point tracking)
    p95_residual: float = 0.0            # 95th percentile of per-tick final residuals
    gamma_hat_median: float = 0.0        # median per-sweep contraction ratio estimate
    mean_lambda: float = 0.0             # mean risk/temperature lambda over the episode


@dataclass
class ExperimentResults:
    """Results from an experiment run."""
    
    metrics: List[EpisodeMetrics]
    mean_return: float
    std_return: float
    mean_steps: float
    std_steps: float
    config: Dict[str, Any] = field(default_factory=dict)