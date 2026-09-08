"""Controlled archive intervention, separate from online lifetime training."""

import numpy as np

from ..agents.context_memory import ContextFieldController
from ..configs.interaction_config import InteractionConfig
from ..envs.command_contact import CommandContactConfig, CommandContactWorld
from .command_contact import acquire, rotated_command


def archive_intervention(seed=71003):
    # Empirical contents from 80 actual source steps. The diagnostic explicitly
    # installs these in one archive and sets its applicability; it does not test
    # recognition or admission (the continuous experiment tests those).
    counts, source = acquire(seed, "aligned")
    rows = []
    for rotation in range(4):
        policies = {}
        for mode in ("intact", "swapped", "erased"):
            agent = ContextFieldController(InteractionConfig(learn=False), seed)
            agent.schema.active[0] = True
            agent.schema.weights[:] = [0, 1, 0, 0]
            agent.schema.archives[0] = counts
            if mode == "swapped":
                agent.schema.archives[0, :, [2, 3]] = agent.schema.archives[0, :, [3, 2]]
            elif mode == "erased":
                agent.schema.archives.fill(0)
            env = CommandContactWorld(CommandContactConfig(seed=seed, rotate=rotation))
            agent.observe(env.reset())
            before = agent.schema.archives.copy()
            agent.think()
            if not np.array_equal(before, agent.schema.archives):
                raise AssertionError("diagnostic imagination changed empirical contents")
            policies[mode] = agent.policy[[rotated_command(a, rotation) for a in (2, 3)]].tolist()
        rows.append({"rotation": rotation, **policies})
    return {
        "source_steps": len(source),
        "source_contacts": sum(r["contact"] for r in source),
        "rows": rows,
        "scope": "Separate diagnostic: acquired contents, explicitly installed archive "
        "and fixed applicability. Same physical scene, fast counts, and mixture weights; "
        "change only archive contents. No diagnostic evidence enters the main experiment.",
    }
