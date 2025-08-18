"""Channel-agnostic potential field composition."""

import numpy as np
from typing import Dict, Optional


def compose_potential(
    attractors: Dict[str, np.ndarray],
    repulsors: Dict[str, np.ndarray], 
    w_attr: Dict[str, float],
    w_rep: Dict[str, float],
    bias: Optional[np.ndarray] = None,
    mode: str = "linear",
    beta_attr: float = 1.0,
    beta_rep: float = 1.0,
    **kwargs
) -> np.ndarray:
    """
    Compose a potential field from arbitrary named channels.
    
    This creates a "weather system" by aggregating multiple pressure systems 
    (attractors and repulsors) with their respective weights.
    
    Args:
        attractors: Dictionary of attractor fields (channel_name -> field)
        repulsors: Dictionary of repulsor fields (channel_name -> field)
        w_attr: Weights for each attractor channel
        w_rep: Weights for each repulsor channel
        bias: Optional bias field to add (e.g., schema)
        mode: Aggregation mode - "linear", "lse" (log-sum-exp), or "maxplus"
        beta_attr: Temperature for LSE aggregation of attractors
        beta_rep: Temperature for LSE aggregation of repulsors
        
    Returns:
        Composed potential field as float32 array
    """
    # Initialize potential field with zeros
    if attractors:
        shape = next(iter(attractors.values())).shape
    elif repulsors:
        shape = next(iter(repulsors.values())).shape
    else:
        raise ValueError("Must provide at least one attractor or repulsor field")
    
    # Allow independent modes via 'mode_attr'/'mode_rep' overrides
    mode_attr = kwargs.get("mode_attr", mode)
    mode_rep = kwargs.get("mode_rep", mode)
    
    # Aggregate attractors based on mode
    if attractors:
        weighted_attrs = []
        for channel_name, field in attractors.items():
            weight = float(w_attr.get(channel_name, 0.0))
            if weight != 0.0:
                weighted_attrs.append(weight * field.astype(np.float32))
        
        if weighted_attrs:
            if mode_attr == "linear":
                P_attr = np.sum(weighted_attrs, axis=0)
            elif mode_attr == "lse":  # Stabilized log-sum-exp
                stack = np.stack(weighted_attrs, axis=0)
                m = np.max(beta_attr * stack, axis=0)
                P_attr = (m + (1.0 / beta_attr) * 
                         np.log(np.sum(np.exp(beta_attr * stack - m[np.newaxis, :, :]), axis=0) + 1e-10))
            elif mode_attr == "maxplus":  # Max-plus algebra
                P_attr = np.max(weighted_attrs, axis=0)
            else:
                raise ValueError(f"Unknown aggregation mode: {mode_attr}")
        else:
            P_attr = np.zeros(shape, dtype=np.float32)
    else:
        P_attr = np.zeros(shape, dtype=np.float32)
    
    # Aggregate repulsors based on mode
    if repulsors:
        weighted_reps = []
        for channel_name, field in repulsors.items():
            weight = float(w_rep.get(channel_name, 0.0))
            if weight != 0.0:
                weighted_reps.append(weight * field.astype(np.float32))
        
        if weighted_reps:
            if mode_rep == "linear":
                P_rep = np.sum(weighted_reps, axis=0)
            elif mode_rep == "lse":  # Stabilized log-sum-exp for repulsors
                stack = np.stack(weighted_reps, axis=0)
                m = np.max(beta_rep * stack, axis=0)
                P_rep = (m + (1.0 / beta_rep) *
                        np.log(np.sum(np.exp(beta_rep * stack - m[np.newaxis, :, :]), axis=0) + 1e-10))
            elif mode_rep == "maxplus":  # Max-plus for strongest repulsor
                P_rep = np.max(weighted_reps, axis=0)
            else:
                raise ValueError(f"Unknown aggregation mode: {mode_rep}")
        else:
            P_rep = np.zeros(shape, dtype=np.float32)
    else:
        P_rep = np.zeros(shape, dtype=np.float32)
    
    # Combine attractors and repulsors
    P = P_attr - P_rep
    
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