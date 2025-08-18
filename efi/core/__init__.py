"""Core utilities for EFI."""

from .diffusion import diffuse_masked
from .fields import (
    update_visit_trail,
    update_novelty,
    corner_hazard,
    wall_proximity_field,
    compute_reachable_frontier,
    effective_potential,
    pick_action_from_potential,
)
from .potential import compose_potential, gradient_follow
from .utils import set_global_seed, ts, ensure_dir

__all__ = [
    "diffuse_masked",
    "update_visit_trail",
    "update_novelty",
    "corner_hazard",
    "wall_proximity_field",
    "compute_reachable_frontier",
    "effective_potential",
    "pick_action_from_potential",
    "compose_potential",
    "gradient_follow",
    "set_global_seed",
    "ts",
    "ensure_dir",
]