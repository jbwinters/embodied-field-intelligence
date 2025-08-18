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
    beta_rep: float = 1.0
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
    
    # Aggregate attractors based on mode
    if attractors:
        weighted_attrs = []
        for channel_name, field in attractors.items():
            weight = float(w_attr.get(channel_name, 0.0))
            if weight != 0.0:
                weighted_attrs.append(weight * field.astype(np.float32))
        
        if weighted_attrs:
            if mode == "linear":
                P_attr = np.sum(weighted_attrs, axis=0)
            elif mode == "lse":  # Log-sum-exp (soft max)
                # LSE(z) = (1/β) * log(Σ exp(β * z_i))
                exp_terms = [np.exp(beta_attr * wa) for wa in weighted_attrs]
                P_attr = (1.0 / beta_attr) * np.log(np.sum(exp_terms, axis=0) + 1e-10)
            elif mode == "maxplus":  # Max-plus algebra
                P_attr = np.max(weighted_attrs, axis=0)
            else:
                raise ValueError(f"Unknown aggregation mode: {mode}")
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
            if mode == "linear":
                P_rep = np.sum(weighted_reps, axis=0)
            elif mode == "lse":  # Log-sum-exp for repulsors
                exp_terms = [np.exp(beta_rep * wr) for wr in weighted_reps]
                P_rep = (1.0 / beta_rep) * np.log(np.sum(exp_terms, axis=0) + 1e-10)
            elif mode == "maxplus":  # Max-plus for strongest repulsor
                P_rep = np.max(weighted_reps, axis=0)
            else:
                raise ValueError(f"Unknown aggregation mode: {mode}")
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