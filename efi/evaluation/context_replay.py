"""Replay schema learning on immutable sensory-derived source records.

No additional physical experience is created. This isolates memory changes
and supports capacity experiments without repeating identical field transport.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from ..agents.context_memory import ContextMemory, ContextMemoryConfig
from ..agents.interaction_schema import InteractionSchema
from ..core.experience import Experience
from .context_memory import scored, memory_info


def replay(stream, mode="bank", cfg=None):
    if mode in ("fast", "slow"):
        schema = InteractionSchema(retention=0.95 if mode == "fast" else 0.995)
    else:
        schema = ContextMemory(cfg or ContextMemoryConfig(), mode=mode)
    observer = SimpleNamespace(schema=schema)
    rows = []
    for row in stream["rows"]:
        data = row["experience"]
        if data is None:
            continue
        exp = Experience(**data)
        p = schema.table()[exp.context, exp.action].astype(np.float64)
        p *= np.asarray(exp.probabilities) > 0
        p /= max(p.sum(), 1e-30)
        exp = replace(exp, probabilities=tuple(p), model_version=schema.version)
        if isinstance(schema, ContextMemory):
            schema.capture(exp)
        feedback = {k: tuple(row[k]) for k in ("displacement", "object_displacement")}
        result = scored(exp, feedback)
        loss, complete = schema.update(exp, feedback["displacement"], row["observed_object_after"])
        assert complete and np.isclose(loss, result["joint_loss"])
        rows.append(
            {
                "tick": row["tick"],
                "phase": row["phase"],
                "contact": row["contact"],
                "contact_index": row["contact_index"],
                "informative_index": row["informative_index"],
                "neutral": row["neutral"],
                "returning": row["returning"],
                "wrong": result["wrong"],
                "joint_loss": loss,
                "memory": memory_info(observer),
            }
        )
    return rows, schema
