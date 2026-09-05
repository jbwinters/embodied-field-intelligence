"""Reproducible CPU latency and fresh-process peak-memory measurements.

Run through ``cli.py interaction-profile`` with BLAS/OMP threads set to one.
Memory includes retained rollout arrays, transport scratch, and Python
allocations. The stress case fills every model-cache site, even though the
two-step evaluation episodes only populate a small neighborhood.
"""

import argparse
from collections import deque
import gc
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tracemalloc

import numpy as np

from ..envs.interaction_world import InteractionWorld, InteractionWorldConfig
from .interaction import acquire, make_agent, run_episode


def array_bytes(value, seen=None):
    """Count unique NumPy backing allocations reachable from an agent."""
    seen = set() if seen is None else seen
    if isinstance(value, np.ndarray):
        while isinstance(value.base, np.ndarray):
            value = value.base
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(array_bytes(v, seen) for v in value.values())
    if isinstance(value, (list, tuple, deque)):
        return sum(array_bytes(v, seen) for v in value)
    if hasattr(value, "__dict__") and value.__class__.__module__.startswith("efi."):
        return array_bytes(vars(value), seen)
    return 0


def memory_worker():
    gc.collect()
    # Linux RSS after imports, before the first agent allocation.
    before = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    tracemalloc.start()
    agent = make_agent(7007)
    env = InteractionWorld(InteractionWorldConfig())
    for _ in range(40):
        run_episode(env, agent)
    normal_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.reset_peak()
    for _ in range(40):
        env.reset()
        agent.reset()
        agent.observe(env.observation())
        agent.rules.values[:] = agent.schema.table()
        agent.rules.versions.fill(agent.schema.version)
        agent.think()
        action = agent.select_action()
        obs, _, _, info = env.step(action)
        agent.after_env_step(info["displacement"])
        agent.observe(obs)
        agent.think()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "normal_peak_allocated_bytes": normal_peak,
        "saturated_cache_peak_allocated_bytes": peak,
        "saturated_cache_retained_bytes": current,
        "agent_unique_array_bytes": array_bytes(agent),
        "rss_before_agent_bytes": before,
        "peak_incremental_rss_bytes": max(
            0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - before
        ),
        "history_records": len(agent.history),
        "saturated_rule_bytes_copied_last_tick": agent.work["rule_bytes_copied"],
        "gather_elements_last_tick": agent.work["gather_elements"],
        "note": "Fresh Linux process after imports. Allocation includes environment and harness "
        "overhead; RSS includes allocator overhead. Saturated caches are a storage stress test, "
        "not additional learned experience. Timings are measured without tracing.",
    }


def profile_interaction(episodes=400, output=None):
    if episodes < 1:
        raise ValueError("positive episode count required")
    worker = subprocess.run(
        [sys.executable, "-m", "efi.evaluation.interaction_profile", "--memory-worker"],
        capture_output=True,
        text=True,
        check=True,
    )
    memory = json.loads(worker.stdout)
    counts, _ = acquire(7007, "push", 2)
    agent = make_agent(7007)
    agent.schema.counts[:] = counts
    for _ in range(25):
        run_episode(InteractionWorld(InteractionWorldConfig()), agent)
    latency, terms = [], []
    for episode in range(episodes):
        env = InteractionWorld(
            InteractionWorldConfig(
                seed=7007 + episode,
                rotate=episode % 4,
                size=(9, 11, 13)[episode % 3],
                layout=("west", "north", "detour")[episode % 3],
            )
        )
        row = run_episode(env, agent)
        latency.extend(row["latency_ms"])
        terms.extend(row["outcome_terms"])
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "unknown",
    )
    result = {
        "platform": platform.platform(),
        "cpu": cpu,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "threads": {
            key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "episodes": episodes,
        "ticks": len(latency),
        "latency_ms": latency,
        "latency_ms_percentiles": dict(
            zip(("p50", "p95", "p99"), np.percentile(latency, [50, 95, 99]).tolist())
        ),
        "max_outcome_terms": max(terms),
        "memory": memory,
        "timing_scope": "Think, sample action, ingest displacement and next observation, score, "
        "learn, gather, publish and transport. Excludes environment, reset, and rendering.",
    }
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-worker", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(memory_worker()))
