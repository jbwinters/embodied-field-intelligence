"""Tests for membrane fields."""

import numpy as np
import pytest

from efi.core.membrane import (
    peripersonal_field,
    brain_membrane_gate,
    compute_membrane_potential,
    adaptive_membrane_radius,
    corridor_membrane
)


class TestPeripersonalField:
    """Test peripersonal membrane field."""
    
    def test_membrane_around_walls(self):
        """Membrane should form around visible walls."""
        H, W = 10, 10
        walls = np.zeros((H, W), dtype=bool)
        walls[5, :] = True  # Horizontal wall
        seen = np.ones((H, W), dtype=bool)  # All visible
        
        field = peripersonal_field(
            walls,
            seen,
            y=3,
            x=5,
            R_base=2.0
        )
        
        # Should be strong near wall
        assert field[4, 5] > 0  # One cell from wall
        assert field[6, 5] > 0  # Other side of wall
        # Should decay with distance
        assert field[3, 5] < field[4, 5]
        assert field[7, 5] < field[6, 5]
    
    def test_dynamic_radius_with_arousal(self):
        """Membrane radius should expand with arousal."""
        H, W = 15, 15
        walls = np.zeros((H, W), dtype=bool)
        walls[7, 7] = True  # Single wall cell
        seen = np.ones((H, W), dtype=bool)
        
        # Low arousal
        field_low = peripersonal_field(
            walls, seen,
            y=7, x=5,
            R_base=2.0,
            arousal=0.0,
            pain=0.0,
            R_gain_arousal=2.0
        )
        
        # High arousal
        field_high = peripersonal_field(
            walls, seen,
            y=7, x=5,
            R_base=2.0,
            arousal=0.8,
            pain=0.0,
            R_gain_arousal=2.0
        )
        
        # Higher arousal should create larger membrane
        assert field_high.sum() > field_low.sum()
    
    def test_dynamic_radius_with_pain(self):
        """Membrane radius should expand with pain."""
        H, W = 15, 15
        walls = np.zeros((H, W), dtype=bool)
        walls[7, 7] = True
        seen = np.ones((H, W), dtype=bool)
        
        # No pain
        field_no_pain = peripersonal_field(
            walls, seen,
            y=7, x=5,
            R_base=2.0,
            arousal=0.0,
            pain=0.0,
            R_gain_pain=3.0
        )
        
        # High pain
        field_pain = peripersonal_field(
            walls, seen,
            y=7, x=5,
            R_base=2.0,
            arousal=0.0,
            pain=0.7,
            R_gain_pain=3.0
        )
        
        # Pain should expand membrane
        assert field_pain.sum() > field_no_pain.sum()
    
    def test_only_visible_walls(self):
        """Membrane should only form around visible walls."""
        H, W = 10, 10
        walls = np.zeros((H, W), dtype=bool)
        walls[5, :] = True  # Horizontal wall
        
        seen = np.zeros((H, W), dtype=bool)
        seen[:3, :] = True  # Only top part visible
        
        field = peripersonal_field(
            walls, seen,
            y=1, x=5,
            R_base=3.0
        )
        
        # No membrane around unseen walls
        assert field[4, 5] == 0  # Near wall but not seen
        assert field[6, 5] == 0


class TestBrainMembraneGate:
    """Test learning rate gating."""
    
    def test_pain_suppresses_learning(self):
        """Pain should suppress learning rate."""
        base = 0.5
        gated = brain_membrane_gate(
            pain=0.6,
            base_rate=base,
            suppress_factor=0.5
        )
        assert gated < base
        assert gated == pytest.approx(0.35, abs=0.01)
    
    def test_minimum_rate_maintained(self):
        """Should maintain minimum learning rate."""
        gated = brain_membrane_gate(
            pain=1.0,
            base_rate=1.0,
            suppress_factor=1.0,
            min_rate=0.2
        )
        assert gated == 0.2
    
    def test_no_pain_no_suppression(self):
        """No pain means no suppression."""
        base = 0.7
        gated = brain_membrane_gate(
            pain=0.0,
            base_rate=base
        )
        assert gated == base


class TestAdaptiveRadius:
    """Test adaptive membrane radius."""
    
    def test_negative_valence_expands(self):
        """Negative valence should expand membrane."""
        r1 = adaptive_membrane_radius(
            base_radius=2.0,
            affect_valence=0.0,
            affect_arousal=0.0,
            affect_control=1.0
        )
        r2 = adaptive_membrane_radius(
            base_radius=2.0,
            affect_valence=-0.5,
            affect_arousal=0.0,
            affect_control=1.0
        )
        assert r2 > r1
    
    def test_high_arousal_expands(self):
        """High arousal should expand membrane."""
        r1 = adaptive_membrane_radius(
            base_radius=2.0,
            affect_valence=0.0,
            affect_arousal=0.0,
            affect_control=1.0
        )
        r2 = adaptive_membrane_radius(
            base_radius=2.0,
            affect_valence=0.0,
            affect_arousal=0.8,
            affect_control=1.0
        )
        assert r2 > r1
    
    def test_low_control_expands(self):
        """Low control should expand membrane."""
        r1 = adaptive_membrane_radius(
            base_radius=2.0,
            affect_valence=0.0,
            affect_arousal=0.0,
            affect_control=1.0
        )
        r2 = adaptive_membrane_radius(
            base_radius=2.0,
            affect_valence=0.0,
            affect_arousal=0.0,
            affect_control=0.2
        )
        assert r2 > r1
    
    def test_radius_clamped(self):
        """Radius should be clamped to min/max."""
        # Test max clamping
        r_max = adaptive_membrane_radius(
            base_radius=4.0,
            affect_valence=-1.0,
            affect_arousal=1.0,
            affect_control=0.0,
            max_radius=5.0
        )
        assert r_max == 5.0
        
        # Test min clamping
        r_min = adaptive_membrane_radius(
            base_radius=1.0,
            affect_valence=1.0,
            affect_arousal=0.0,
            affect_control=1.0,
            v_weight=-2.0,
            min_radius=0.5
        )
        assert r_min == 0.5


class TestCorridorMembrane:
    """Test corridor-specific membrane."""
    
    def test_corridor_detection(self):
        """Should detect narrow corridors."""
        H, W = 15, 15
        walls = np.ones((H, W), dtype=bool)
        # Create vertical corridor
        walls[5:10, 6:9] = False  # 3-wide corridor
        
        field = corridor_membrane(
            walls,
            corridor_width=4.0,
            strength=1.0
        )
        
        # Should have membrane in corridor
        assert field[7, 7] > 0  # Center of corridor
        # Should be zero in walls
        assert field[0, 0] == 0
        assert field[5, 5] == 0
    
    def test_centering_field(self):
        """Corridor membrane should create centering force."""
        H, W = 10, 10
        walls = np.ones((H, W), dtype=bool)
        # Horizontal corridor
        walls[4:6, :] = False  # 2-wide corridor
        
        field = corridor_membrane(
            walls,
            corridor_width=3.0,
            strength=1.0
        )
        
        # Should be stronger near walls
        assert field[4, 5] > 0
        assert field[5, 5] > 0
        # Center should have some value
        mid_val = (field[4, 5] + field[5, 5]) / 2
        assert mid_val > 0