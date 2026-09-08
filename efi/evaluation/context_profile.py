"""EFI-02 fresh-process allocation and isolated one-worker CPU measurements."""

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter
import tracemalloc

import numpy as np

from ..envs.context_world import ContextWorld, ContextWorldConfig
from .context_memory import make_agent
from .command_profile import percentiles
from .interaction_profile import array_bytes


def memory_worker():
    gc.collect()
    before = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    tracemalloc.start()
    env, agent = ContextWorld(ContextWorldConfig(seed=81000)), make_agent(81000, "bank")
    agent.observe(env.reset())
    for _ in range(env.max_steps):
        agent.think()
        action = agent.select_action()
        obs, _, _, info = env.step(action)
        agent.after_env_step(info["displacement"])
        agent.observe(obs)
    normal_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.reset_peak()
    # Storage stress only; these synthetic copies are never labeled experience.
    env.reset()
    agent.reset()
    agent.observe(env.observation())
    agent.schema.active.fill(True)
    agent.schema.archives[:] = agent.schema.counts
    agent.rules.values[:] = agent.schema.table()
    agent.rules.versions.fill(agent.schema.version)
    for _ in range(40):
        agent.think()
        action = agent.select_action()
        obs, _, _, info = env.step(action)
        agent.after_env_step(info["displacement"])
        agent.observe(obs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "normal_peak_allocated_bytes": normal_peak,
        "saturated_cache_peak_allocated_bytes": peak,
        "saturated_cache_retained_bytes": current,
        "agent_unique_array_bytes": array_bytes(agent),
        "bank_array_bytes": agent.schema.storage_bytes,
        "history_records": len(agent.history),
        "rss_before_agent_bytes": before,
        "peak_incremental_rss_bytes": max(
            0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - before
        ),
        "scope": "Fresh process after imports, full continuous lifetime "
        "including scratch and world. "
        "All archive slots and rule caches then filled as storage stress. "
        "Forty further physical ticks score and update normally; "
        "synthetic saturation copies are not additional observations.",
    }


def profile_context(lifetimes=3, output=None):
    if lifetimes < 1 or any(
        os.environ.get(k) != "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
    ):
        raise ValueError(
            "positive lifetime count and OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 required"
        )
    worker = subprocess.run(
        [sys.executable, "-m", "efi.evaluation.context_profile", "--memory-worker"],
        text=True,
        capture_output=True,
        check=True,
    )
    memory = json.loads(worker.stdout)
    latency, startup = [], []
    for seed in range(81000, 81000 + lifetimes):
        env = ContextWorld(ContextWorldConfig(seed=seed))
        started = perf_counter()
        agent = make_agent(seed, "bank")
        agent.observe(env.reset())
        startup.append(1000 * (perf_counter() - started))
        for _ in range(env.max_steps):
            started = perf_counter()
            agent.think()
            action = agent.select_action()
            elapsed = perf_counter() - started
            obs, _, _, info = env.step(action)
            started = perf_counter()
            agent.after_env_step(info["displacement"])
            agent.observe(obs)
            latency.append(1000 * (elapsed + perf_counter() - started))
    timing = percentiles(latency)
    result = {
        "cpu": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "threads": {k: os.environ[k] for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        "lifetimes": lifetimes,
        "ticks": len(latency),
        "latency_ms": latency,
        "latency_ms_percentiles": timing,
        "constructor_and_observation_ms": startup,
        "memory": memory,
        "resource_gates": {
            "allocation": max(
                memory["normal_peak_allocated_bytes"],
                memory["saturated_cache_peak_allocated_bytes"],
            )
            <= 32 * 2**20,
            "rss": memory["peak_incremental_rss_bytes"] <= 96 * 2**20,
            "p95": timing["p95"] <= 50,
            "p99": timing["p99"] <= 100,
        },
        "scope": "Run alone, one numerical worker, no GPU. "
        "Every lifetime tick, no warmup excluded. "
        "Think, sample, body feedback, observation, score, learn, gather, publish and transport; "
        "exclude environment/serialization, report construction separately. "
        "Allocation tracing in separate worker.",
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
