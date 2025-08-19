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
from .affect import (
    AffectState,
    compute_nociception,
    update_affect,
    pain_to_temperature,
    compute_learning_gate,
    pain_field,
)
from .membrane import (
    peripersonal_field,
    brain_membrane_gate,
    compute_membrane_potential,
    adaptive_membrane_radius,
    corridor_membrane,
)

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
    # Affect system
    "AffectState",
    "compute_nociception",
    "update_affect",
    "pain_to_temperature",
    "compute_learning_gate",
    "pain_field",
    # Membrane system
    "peripersonal_field",
    "brain_membrane_gate",
    "compute_membrane_potential",
    "adaptive_membrane_radius",
    "corridor_membrane",
]