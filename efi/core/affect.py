"""Affect and nociception system for embodied agents."""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class AffectState:
    """
    Affective state of the agent.
    
    Attributes:
        valence: Positive/negative emotional tone (-1 to 1)
        arousal: Activation/energy level (0 to 1)
        control: Sense of agency/control (0 to 1)
        pain: Current nociception level (0 to 1)
    """
    valence: float = 0.0
    arousal: float = 0.0
    control: float = 1.0
    pain: float = 0.0
    
    def to_dict(self):
        """Convert to dictionary for logging."""
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "control": self.control,
            "pain": self.pain
        }


def compute_nociception(
    bump: bool,
    neg_reward: float,
    wall_prox_here: float,
    stuck_count: int,
    bump_weight: float = 0.5,
    reward_weight: float = 0.3,
    prox_weight: float = 0.1,
    stuck_weight: float = 0.1
) -> float:
    """
    Compute nociception (pain) signal from various negative stimuli.
    
    Args:
        bump: Whether agent bumped into wall this step
        neg_reward: Negative reward received (e.g., from B targets)
        wall_prox_here: Wall proximity at current location (0-1)
        stuck_count: Number of consecutive stuck steps
        bump_weight: Weight for bump contribution
        reward_weight: Weight for negative reward contribution
        prox_weight: Weight for wall proximity contribution
        stuck_weight: Weight for stuck contribution
    
    Returns:
        Nociception level (0-1)
    """
    pain = 0.0
    
    # Bump causes immediate pain spike
    if bump:
        pain += bump_weight
    
    # Negative rewards (B pickups) cause pain
    if neg_reward < 0:
        pain += reward_weight * abs(neg_reward)
    
    # Wall proximity causes mild discomfort
    pain += prox_weight * wall_prox_here
    
    # Being stuck causes increasing pain
    if stuck_count > 0:
        stuck_pain = stuck_weight * min(stuck_count / 10.0, 1.0)
        pain += stuck_pain
    
    return np.clip(pain, 0.0, 1.0)


def update_affect(
    state: AffectState,
    nociception: float,
    surprise: float,
    reward: float,
    rho_v: float = 0.02,
    rho_a: float = 0.05,
    rho_c: float = 0.05,
    rho_p: float = 0.1
) -> AffectState:
    """
    Update affective state based on current stimuli.
    
    Uses exponentially weighted moving average (EWMA) for smooth transitions.
    
    Args:
        state: Current affect state
        nociception: Current pain level (0-1)
        surprise: Prediction error or novelty (0-1)
        reward: Immediate reward received
        rho_v: EWMA decay rate for valence
        rho_a: EWMA decay rate for arousal
        rho_c: EWMA decay rate for control
        rho_p: EWMA decay rate for pain
    
    Returns:
        Updated affect state
    """
    # Update pain with EWMA
    new_pain = (1 - rho_p) * state.pain + rho_p * nociception
    
    # Valence: positive rewards increase, pain decreases
    target_valence = np.tanh(reward - new_pain)
    new_valence = (1 - rho_v) * state.valence + rho_v * target_valence
    
    # Arousal: increases with pain, surprise, and absolute reward
    target_arousal = np.clip(new_pain + 0.5 * surprise + 0.3 * abs(reward), 0, 1)
    new_arousal = (1 - rho_a) * state.arousal + rho_a * target_arousal
    
    # Control: decreases with pain and stuck situations
    target_control = max(0, 1.0 - new_pain - 0.5 * max(0, -reward))
    new_control = (1 - rho_c) * state.control + rho_c * target_control
    
    return AffectState(
        valence=np.clip(new_valence, -1, 1),
        arousal=np.clip(new_arousal, 0, 1),
        control=np.clip(new_control, 0, 1),
        pain=np.clip(new_pain, 0, 1)
    )


def pain_to_temperature(
    base_temp: float,
    pain: float,
    arousal: float,
    gain: float = 0.6,
    max_temp: float = 2.0
) -> float:
    """
    Convert pain and arousal to action temperature adjustment.
    
    Higher pain increases temperature to enable escape behaviors.
    
    Args:
        base_temp: Base temperature from other sources
        pain: Current pain level (0-1)
        arousal: Current arousal level (0-1)
        gain: Conversion gain factor
        max_temp: Maximum allowed temperature
    
    Returns:
        Adjusted temperature
    """
    # Pain and arousal both increase temperature
    pain_boost = gain * (pain + 0.3 * arousal)
    new_temp = base_temp + pain_boost
    
    return min(new_temp, max_temp)


def compute_learning_gate(
    pain: float,
    base_rate: float,
    pain_suppress: float = 0.5,
    min_rate: float = 0.1
) -> float:
    """
    Compute learning rate gate based on pain (brain membrane).
    
    High pain suppresses learning to prevent maladaptive associations.
    
    Args:
        pain: Current pain level (0-1)
        base_rate: Base learning rate
        pain_suppress: Pain suppression factor
        min_rate: Minimum learning rate (never fully stop)
    
    Returns:
        Gated learning rate
    """
    # Suppress learning under high pain
    gate = max(min_rate, 1.0 - pain_suppress * pain)
    return base_rate * gate


def pain_field(
    pain: float,
    y: int,
    x: int,
    H: int,
    W: int,
    radius: float = 2.0,
    decay: float = 0.5
) -> np.ndarray:
    """
    Generate a pain field centered at agent location.
    
    Creates a repulsive field that pushes the agent away from
    painful locations.
    
    Args:
        pain: Current pain level (0-1)
        y, x: Agent position
        H, W: Field dimensions
        radius: Pain field radius
        decay: Spatial decay rate
    
    Returns:
        Pain field array (H, W)
    """
    field = np.zeros((H, W), dtype=np.float32)
    
    if pain > 0:
        # Create distance map from agent position
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        dist = np.sqrt((yy - y)**2 + (xx - x)**2)
        
        # Pain field decays with distance
        mask = dist <= radius
        field[mask] = pain * np.exp(-decay * dist[mask])
    
    return field