"""Tests for affect and nociception system."""

import numpy as np
import pytest

from efi.core.affect import (
    AffectState,
    compute_nociception,
    update_affect,
    pain_to_temperature,
    pain_field
)
from efi.core.membrane import brain_membrane_gate


class TestNociception:
    """Test nociception computation."""
    
    def test_bump_causes_pain(self):
        """Bumping into walls should cause pain."""
        pain = compute_nociception(
            bump=True,
            neg_reward=0,
            wall_prox_here=0,
            stuck_count=0,
            bump_weight=0.5
        )
        assert pain > 0
        assert pain == 0.5
    
    def test_negative_reward_causes_pain(self):
        """Negative rewards should cause pain."""
        pain = compute_nociception(
            bump=False,
            neg_reward=-1.0,
            wall_prox_here=0,
            stuck_count=0,
            reward_weight=0.3
        )
        assert pain > 0
        assert pain == 0.3
    
    def test_wall_proximity_causes_pain(self):
        """Being near walls should cause mild pain."""
        pain = compute_nociception(
            bump=False,
            neg_reward=0,
            wall_prox_here=0.5,
            stuck_count=0,
            prox_weight=0.1
        )
        assert pain > 0
        assert pain == 0.05
    
    def test_stuck_causes_increasing_pain(self):
        """Being stuck should cause increasing pain."""
        pain1 = compute_nociception(
            bump=False,
            neg_reward=0,
            wall_prox_here=0,
            stuck_count=1,
            stuck_weight=0.1
        )
        pain5 = compute_nociception(
            bump=False,
            neg_reward=0,
            wall_prox_here=0,
            stuck_count=5,
            stuck_weight=0.1
        )
        assert pain5 > pain1
        assert pain1 > 0
    
    def test_pain_is_clamped(self):
        """Pain should be clamped to [0, 1]."""
        pain = compute_nociception(
            bump=True,
            neg_reward=-5.0,
            wall_prox_here=1.0,
            stuck_count=100,
            bump_weight=1.0,
            reward_weight=1.0,
            prox_weight=1.0,
            stuck_weight=1.0
        )
        assert pain == 1.0


class TestAffectState:
    """Test affect state updates."""
    
    def test_pain_decreases_valence(self):
        """Pain should decrease valence."""
        state = AffectState()
        new_state = update_affect(
            state,
            nociception=0.8,
            surprise=0,
            reward=0,
            rho_v=1.0  # Immediate update
        )
        assert new_state.valence < 0
    
    def test_reward_increases_valence(self):
        """Positive reward should increase valence."""
        state = AffectState()
        new_state = update_affect(
            state,
            nociception=0,
            surprise=0,
            reward=1.0,
            rho_v=1.0  # Immediate update
        )
        assert new_state.valence > 0
    
    def test_pain_increases_arousal(self):
        """Pain should increase arousal."""
        state = AffectState()
        new_state = update_affect(
            state,
            nociception=0.8,
            surprise=0,
            reward=0,
            rho_a=1.0,  # Immediate arousal update
            rho_p=1.0   # Immediate pain update for testing
        )
        assert new_state.arousal > 0
        # With immediate pain update (rho_p=1.0), pain=0.8
        # Arousal = clamp(pain + 0.5*surprise + 0.3*|reward|) = 0.8
        assert new_state.arousal >= 0.8
    
    def test_pain_decreases_control(self):
        """Pain should decrease control."""
        state = AffectState(control=1.0)
        new_state = update_affect(
            state,
            nociception=0.8,
            surprise=0,
            reward=0,
            rho_c=1.0  # Immediate update
        )
        assert new_state.control < 1.0
    
    def test_ewma_smoothing(self):
        """EWMA should smooth transitions."""
        state = AffectState()
        # Apply high pain with low decay rate
        new_state = update_affect(
            state,
            nociception=1.0,
            surprise=0,
            reward=0,
            rho_p=0.1  # Slow update
        )
        assert 0 < new_state.pain < 1.0  # Should be between 0 and 1
        assert new_state.pain == pytest.approx(0.1, abs=0.01)
    
    def test_pain_decay(self):
        """Pain should decay over time without stimuli."""
        state = AffectState(pain=0.8)
        new_state = update_affect(
            state,
            nociception=0,
            surprise=0,
            reward=0,
            rho_p=0.1
        )
        assert new_state.pain < state.pain
        assert new_state.pain == pytest.approx(0.72, abs=0.01)


class TestTemperatureModulation:
    """Test pain-based temperature adjustment."""
    
    def test_pain_increases_temperature(self):
        """Pain should increase action temperature."""
        base_temp = 0.5
        new_temp = pain_to_temperature(
            base_temp,
            pain=0.5,
            arousal=0,
            gain=0.6
        )
        assert new_temp > base_temp
        assert new_temp == pytest.approx(0.8, abs=0.01)
    
    def test_arousal_increases_temperature(self):
        """Arousal should also increase temperature."""
        base_temp = 0.5
        new_temp = pain_to_temperature(
            base_temp,
            pain=0,
            arousal=0.5,
            gain=0.6
        )
        assert new_temp > base_temp
    
    def test_temperature_is_capped(self):
        """Temperature should have a maximum."""
        base_temp = 1.0
        new_temp = pain_to_temperature(
            base_temp,
            pain=1.0,
            arousal=1.0,
            gain=2.0,
            max_temp=2.0
        )
        assert new_temp == 2.0


class TestLearningGate:
    """Test brain membrane learning gate."""
    
    def test_pain_suppresses_learning(self):
        """High pain should suppress learning."""
        base_rate = 1.0
        gated_rate = brain_membrane_gate(
            pain=0.8,
            base_rate=base_rate,
            suppress_factor=0.5,
            min_rate=0.1
        )
        assert gated_rate < base_rate
        assert gated_rate >= 0.1
    
    def test_minimum_learning_maintained(self):
        """Learning should never fully stop."""
        base_rate = 1.0
        gated_rate = brain_membrane_gate(
            pain=1.0,
            base_rate=base_rate,
            suppress_factor=1.0,
            min_rate=0.1
        )
        assert gated_rate == 0.1
    
    def test_no_pain_no_suppression(self):
        """No pain means no suppression."""
        base_rate = 0.5
        gated_rate = brain_membrane_gate(
            pain=0,
            base_rate=base_rate,
            suppress_factor=0.5,
            min_rate=0.1
        )
        assert gated_rate == base_rate


class TestPainField:
    """Test pain field generation."""
    
    def test_pain_field_centered_at_agent(self):
        """Pain field should be centered at agent location."""
        H, W = 10, 10
        y, x = 5, 5
        field = pain_field(
            pain=1.0,
            y=y,
            x=x,
            H=H,
            W=W,
            radius=2.0
        )
        # Maximum at agent location
        assert field[y, x] == field.max()
        # Decays with distance
        assert field[y+1, x] < field[y, x]
        assert field[y, x+1] < field[y, x]
    
    def test_no_pain_no_field(self):
        """No pain means no field."""
        H, W = 10, 10
        field = pain_field(
            pain=0,
            y=5,
            x=5,
            H=H,
            W=W
        )
        assert field.sum() == 0
    
    def test_pain_field_respects_radius(self):
        """Pain field should respect radius."""
        H, W = 20, 20
        y, x = 10, 10
        radius = 3.0
        field = pain_field(
            pain=1.0,
            y=y,
            x=x,
            H=H,
            W=W,
            radius=radius
        )
        # Should be zero outside radius
        assert field[y+5, x] == 0
        assert field[y, x+5] == 0
        # Should be non-zero inside radius
        assert field[y+2, x] > 0
        assert field[y, x+2] > 0