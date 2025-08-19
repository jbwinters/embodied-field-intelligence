"""Protective membrane fields for safer navigation."""

import numpy as np
from scipy.ndimage import distance_transform_edt


def peripersonal_field(
    known_walls: np.ndarray,
    seen: np.ndarray,
    y: int,
    x: int,
    R_base: float,
    arousal: float = 0.0,
    pain: float = 0.0,
    R_gain_arousal: float = 1.0,
    R_gain_pain: float = 1.5,
    max_radius: float = 5.0
) -> np.ndarray:
    """
    Generate peripersonal (body) membrane field.
    
    Creates a repulsive field around walls that expands with arousal/pain,
    helping the agent maintain safe distance from obstacles.
    
    Args:
        known_walls: Binary array of known wall locations
        seen: Binary array of seen locations
        y, x: Agent position
        R_base: Base membrane radius
        arousal: Current arousal level (0-1)
        pain: Current pain level (0-1)
        R_gain_arousal: Arousal contribution to radius
        R_gain_pain: Pain contribution to radius
        max_radius: Maximum allowed radius
    
    Returns:
        Membrane field array (same shape as known_walls)
    """
    H, W = known_walls.shape
    field = np.zeros((H, W), dtype=np.float32)
    
    # Dynamic radius based on affective state
    R_t = R_base + R_gain_arousal * arousal + R_gain_pain * pain
    R_t = np.clip(R_t, R_base, max_radius)
    
    # Only create membrane for walls we can see
    visible_walls = known_walls & seen
    
    if not visible_walls.any():
        return field
    
    # Distance transform from walls
    dist_from_walls = distance_transform_edt(~visible_walls)
    
    # Create membrane field that decays with distance
    # Stronger near walls, decays to 0 at radius R_t
    mask = dist_from_walls < R_t
    field[mask] = (1.0 - dist_from_walls[mask] / R_t) ** 2
    
    # Boost membrane strength near agent when in pain
    if pain > 0.1:
        # Local boost around agent position
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        agent_dist = np.sqrt((yy - y)**2 + (xx - x)**2)
        local_boost = pain * np.exp(-0.5 * agent_dist)
        field = np.maximum(field, local_boost * field)
    
    return field


def brain_membrane_gate(
    pain: float,
    base_rate: float,
    suppress_factor: float = 0.5,
    min_rate: float = 0.1
) -> float:
    """
    Compute learning rate gate based on pain (brain membrane).
    
    High pain suppresses plasticity to prevent maladaptive learning.
    
    Args:
        pain: Current pain level (0-1)
        base_rate: Base learning rate
        suppress_factor: How much pain suppresses learning
        min_rate: Minimum learning rate (never fully stop)
    
    Returns:
        Gated learning rate
    """
    # Suppress learning proportional to pain
    gate = max(min_rate, 1.0 - suppress_factor * pain)
    return base_rate * gate


def compute_membrane_potential(
    membrane_field: np.ndarray,
    weight: float = 0.6
) -> np.ndarray:
    """
    Convert membrane field to potential for composition.
    
    Args:
        membrane_field: Raw membrane field (0-1)
        weight: Membrane weight
    
    Returns:
        Weighted membrane potential
    """
    return weight * membrane_field


def adaptive_membrane_radius(
    base_radius: float,
    affect_valence: float,
    affect_arousal: float,
    affect_control: float,
    v_weight: float = -0.3,
    a_weight: float = 1.0,
    c_weight: float = -0.5,
    min_radius: float = 0.5,
    max_radius: float = 5.0
) -> float:
    """
    Compute adaptive membrane radius based on full affective state.
    
    - Negative valence expands membrane (more cautious)
    - High arousal expands membrane (more reactive)
    - Low control expands membrane (less agency)
    
    Args:
        base_radius: Base membrane radius
        affect_valence: Current valence (-1 to 1)
        affect_arousal: Current arousal (0 to 1)
        affect_control: Current control (0 to 1)
        v_weight: Valence contribution weight
        a_weight: Arousal contribution weight
        c_weight: Control contribution weight (negative = inverse)
        min_radius: Minimum radius
        max_radius: Maximum radius
    
    Returns:
        Adapted radius
    """
    # Compute adjustments
    v_adjust = v_weight * affect_valence  # Negative valence increases radius (v_weight should be negative)
    a_adjust = a_weight * affect_arousal   # High arousal increases radius
    c_adjust = -c_weight * (1.0 - affect_control)  # Low control increases radius (note the sign)
    
    # Apply adjustments
    radius = base_radius + v_adjust + a_adjust + c_adjust
    
    return np.clip(radius, min_radius, max_radius)


def corridor_membrane(
    walls: np.ndarray,
    corridor_width: float = 3.0,
    strength: float = 1.0
) -> np.ndarray:
    """
    Special membrane for narrow corridors.
    
    Creates centering force in tight spaces.
    
    Args:
        walls: Binary wall array
        corridor_width: Width threshold for corridor detection
        strength: Membrane strength in corridors
    
    Returns:
        Corridor membrane field
    """
    H, W = walls.shape
    field = np.zeros((H, W), dtype=np.float32)
    
    # Distance from walls
    dist = distance_transform_edt(~walls)
    
    # Detect corridor regions (narrow passages)
    # A point is in a corridor if it's free space with limited distance to walls
    corridor_mask = (dist > 0) & (dist <= corridor_width)
    
    if corridor_mask.any():
        # In corridors, create centering field
        # Field is stronger near walls, weaker in center
        # This creates a "push away from walls" effect
        max_dist = corridor_width
        field[corridor_mask] = strength * (1.0 - dist[corridor_mask] / max_dist) ** 2
    
    return field