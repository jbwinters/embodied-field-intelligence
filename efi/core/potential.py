"""Channel-agnostic potential field composition."""

import numpy as np
from typing import Dict, Optional


def compose_potential(
    attractors: Dict[str, np.ndarray],
    repulsors: Dict[str, np.ndarray], 
    w_attr: Dict[str, float],
    w_rep: Dict[str, float],
    bias: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compose a potential field from arbitrary named channels.
    
    This creates a "weather system" by linearly superposing multiple
    pressure systems (attractors and repulsors) with their respective weights.
    
    Args:
        attractors: Dictionary of attractor fields (channel_name -> field)
        repulsors: Dictionary of repulsor fields (channel_name -> field)
        w_attr: Weights for each attractor channel
        w_rep: Weights for each repulsor channel
        bias: Optional bias field to add (e.g., schema)
        
    Returns:
        Composed potential field as float32 array
    """
    # Initialize potential field with zeros
    if attractors:
        P = np.zeros_like(next(iter(attractors.values())), dtype=np.float32)
    elif repulsors:
        P = np.zeros_like(next(iter(repulsors.values())), dtype=np.float32)
    else:
        raise ValueError("Must provide at least one attractor or repulsor field")
    
    # Add weighted attractors (positive contribution)
    for channel_name, field in attractors.items():
        weight = float(w_attr.get(channel_name, 0.0))
        if weight != 0.0:
            P = P + weight * field.astype(np.float32)
    
    # Subtract weighted repulsors (negative contribution)
    for channel_name, field in repulsors.items():
        weight = float(w_rep.get(channel_name, 0.0))
        if weight != 0.0:
            P = P - weight * field.astype(np.float32)
    
    # Add optional bias field
    if bias is not None:
        P = P + bias.astype(np.float32)
    
    return P


def gradient_follow(P: np.ndarray, y: int, x: int) -> tuple[float, float]:
    """
    Extract gradient at a position for continuous control.
    
    Args:
        P: Potential field
        y, x: Current position
        
    Returns:
        Gradient components (gy, gx) for gradient following
    """
    gy, gx = np.gradient(P.astype(np.float32))
    return float(gy[y, x]), float(gx[y, x])