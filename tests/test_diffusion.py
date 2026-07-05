"""Tests for diffusion operations."""

import numpy as np
import pytest

from efi.core import diffuse_masked


class TestDiffusion:
    """Test diffusion operations."""
    
    def test_diffuse_masked_basic(self):
        """Test basic diffusion without walls."""
        # Create a field with a single hot spot
        field = np.zeros((5, 5), dtype=np.float32)
        field[2, 2] = 1.0
        walls = np.zeros_like(field, dtype=bool)
        
        # Diffuse
        result = diffuse_masked(field, walls, diff=0.5, decay=0.0, steps=1)
        
        # Check that value spread to neighbors
        assert result[2, 2] < 1.0  # Center decreased
        assert result[1, 2] > 0.0  # Top neighbor increased
        assert result[3, 2] > 0.0  # Bottom neighbor increased
        assert result[2, 1] > 0.0  # Left neighbor increased
        assert result[2, 3] > 0.0  # Right neighbor increased
    
    def test_diffuse_with_walls(self):
        """Test diffusion with wall blocking."""
        # Create field with walls
        field = np.zeros((5, 5), dtype=np.float32)
        field[2, 2] = 1.0
        walls = np.zeros_like(field, dtype=bool)
        walls[2, 3] = True  # Wall to the right
        
        # Diffuse
        result = diffuse_masked(field, walls, diff=0.5, decay=0.0, steps=1)
        
        # Check that wall blocks diffusion
        assert result[2, 3] == 0.0  # Wall cell remains zero
        assert result[2, 1] > 0.0   # Left neighbor still gets diffusion
    
    def test_diffuse_with_decay(self):
        """Test diffusion with decay."""
        # Create uniform field
        field = np.ones((5, 5), dtype=np.float32)
        walls = np.zeros_like(field, dtype=bool)
        
        # Diffuse with decay
        result = diffuse_masked(field, walls, diff=0.0, decay=0.1, steps=1)
        
        # Check that all values decreased
        assert np.all(result < 1.0)
        assert np.allclose(result, 0.9)
    
    def test_boundary_conditions(self):
        """Grid edges are zero-flux, not sinks: scent on border cells must
        survive diffusion (agents can walk on edges), while wall cells are
        forced to zero."""
        field = np.ones((5, 5), dtype=np.float32)
        walls = np.zeros_like(field, dtype=bool)
        walls[2, 2] = True

        result = diffuse_masked(field, walls, diff=0.1, decay=0.0, steps=1)

        # Border cells keep their scent (edge-safe averaging, no leakage)
        assert np.all(result[0, :] > 0.0)   # Top edge
        assert np.all(result[-1, :] > 0.0)  # Bottom edge
        assert np.all(result[:, 0] > 0.0)   # Left edge
        assert np.all(result[:, -1] > 0.0)  # Right edge

        # Walls act as sinks
        assert result[2, 2] == 0.0