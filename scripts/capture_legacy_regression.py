"""Capture the existing 200 published-config and 12 egocentric regression episodes."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from efi.agents import EgocentricFieldController
from efi.configs import Ablations, AgentConfig, EnvConfig
from efi.envs import ForageWorld
from efi.evaluation import run_ego_episode, run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    result = run_experiment(
        EnvConfig(),
        AgentConfig(valA_init=1.0),
        None,
        Ablations(schema=0),
        episodes=40,
        seeds=5,
        use_controller=True,
    )
    rows = {"published_200": [asdict(m) for m in result.metrics], "ego_12": []}
    for seed in range(12):
        env = ForageWorld(EnvConfig(H=15, W=15, seed=seed))
        agent = EgocentricFieldController(
            AgentConfig(valA_init=1.0),
            Ablations(schema=0),
            seed=seed,
        )
        _, metrics = run_ego_episode(env, agent)
        rows["ego_12"].append(asdict(metrics))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, sort_keys=True))
    print(hashlib.sha256(args.out.read_bytes()).hexdigest(), args.out)


if __name__ == "__main__":
    main()
