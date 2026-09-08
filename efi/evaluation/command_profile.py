"""EFI-01 CPU and fresh-process memory accounting; no benchmark overlap."""

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

from ..envs.command_contact import CommandContactConfig, CommandContactWorld
from .command_contact import acquire, common_transition, make_agent, run_target, LAWS, LAYOUTS
from .interaction import CONTEXTS
from .interaction_profile import array_bytes


def memory_worker():
    gc.collect()
    before = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    tracemalloc.start()
    agent = make_agent(41000)
    for rep in range(2):
        for context in CONTEXTS:
            for command in range(5):
                env = CommandContactWorld(
                    CommandContactConfig(
                        seed=41000 + rep, source=True, context=context, max_steps=1
                    )
                )
                common_transition(env, {"conditioned": agent}, command)
    acquisition_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.reset_peak()
    for i in range(40):
        run_target(
            CommandContactWorld(
                CommandContactConfig(
                    seed=41000 + i, rotate=i % 4, layout=LAYOUTS[i % 2], size=(9, 11, 13)[i % 3]
                )
            ),
            agent,
        )
    normal_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.reset_peak()
    for i in range(40):
        env = CommandContactWorld(CommandContactConfig(seed=41000 + i, rotate=i % 4))
        agent.reset()
        agent.observe(env.reset())
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
        "acquisition_peak_allocated_bytes": acquisition_peak,
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
        "scope": "Fresh process after imports; one learner including scratch, "
        "environment and harness. "
        "80 real acquisition transitions, 40 target trials, 40 saturated-cache iterations. "
        "Saturation is storage stress, never additional empirical evidence.",
    }


def percentiles(values):
    return dict(zip(("p50", "p95", "p99"), np.percentile(values, [50, 95, 99]).tolist()))


def profile_command(episodes=400, output=None):
    if episodes < 2:
        raise ValueError("at least two trials required to profile both laws")
    if any(os.environ.get(k) != "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")):
        raise ValueError("set OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 before launching Python")
    worker = subprocess.run(
        [sys.executable, "-m", "efi.evaluation.command_profile", "--memory-worker"],
        capture_output=True,
        text=True,
        check=True,
    )
    memory = json.loads(worker.stdout)
    latency, startup, constructors, acquisition, terms, rule_bytes = [], [], [], [], [], []
    for law_index, law in enumerate(LAWS):
        counts, source = acquire(41000 + law_index, law)
        acquisition.extend(r["latency_ms"]["conditioned"] for r in source)
        started = perf_counter()
        agent = make_agent(41000 + law_index, counts=counts)
        constructors.append(1000 * (perf_counter() - started))
        for i in range(25):
            run_target(CommandContactWorld(CommandContactConfig(seed=41000 + i, rule=law)), agent)
        for i in range(law_index, episodes, 2):
            row = run_target(
                CommandContactWorld(
                    CommandContactConfig(
                        seed=41000 + i,
                        rule=law,
                        rotate=(i // 2) % 4,
                        layout=LAYOUTS[(i // 2) % 2],
                        front_wall=bool((i // 4) % 2),
                        size=(9, 11, 13)[i % 3],
                    )
                ),
                agent,
            )
            startup.append(row["startup_ms"])
            latency.extend(t["latency_ms"] for t in row["transitions"])
            terms.extend(t["outcome_terms"] for t in row["transitions"])
            rule_bytes.extend(t["rule_bytes_copied"] for t in row["transitions"])
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "unknown",
    )
    timing = percentiles(latency)
    peak = max(
        memory[k]
        for k in (
            "acquisition_peak_allocated_bytes",
            "normal_peak_allocated_bytes",
            "saturated_cache_peak_allocated_bytes",
        )
    )
    result = {
        "cpu": cpu,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "threads": {k: os.environ[k] for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        "episodes": episodes,
        "ticks": len(latency),
        "warmup_trials_per_law": 25,
        "latency_ms": latency,
        "latency_ms_percentiles": timing,
        "constructor_ms": constructors,
        "reset_and_first_observation_ms": percentiles(startup),
        "source_transition_including_reset_ms": percentiles(acquisition),
        "max_outcome_terms": max(terms),
        "max_rule_bytes_copied": max(rule_bytes),
        "memory": memory,
        "resource_gates": {
            "allocation": peak <= 32 * 2**20,
            "rss": memory["peak_incremental_rss_bytes"] <= 96 * 2**20,
            "p95": timing["p95"] <= 50,
            "p99": timing["p99"] <= 100,
        },
        "timing_scope": "One numerical worker, no allocation tracing during timing. "
        "Think, sample, ingest displacement and observation, score, learn, gather, publish "
        "and transport. Target excludes environment, resets, recording and result "
        "serialization; startup and source reset costs are separate.",
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
