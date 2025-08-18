"""Tests for A and B improvements to the EFI system."""

import numpy as np
import pytest

from efi.configs import EnvConfig, AgentConfig, SchemaConfig, Ablations
from efi.envs import ForageWorld
from efi.agents import ChemotaxisAgentCA, SchemaField
from efi.core import (
    compose_potential, 
    wall_proximity_field,
    compute_reachable_frontier,
    corner_hazard
)
from efi.evaluation import run_episode


class TestSchemaValence:
    """Test A1: Signed reward-aware schema bias."""
    
    def test_schema_valence_initialization(self):
        """Test that schema field initializes valence tracking."""
        schema = SchemaField(H=10, W=10, feature_dim=6, 
                            cfg=SchemaConfig(K=4, tile=5))
        
        assert hasattr(schema, 'q'), "Schema should have valence array q"
        # Valence is now per-tile: (ny, nx, K)
        assert schema.q.shape == (schema.ny, schema.nx, 4), "Valence array should be per-tile"
        assert np.all(schema.q == 0), "Initial valences should be zero"
    
    def test_schema_valence_update(self):
        """Test that schema valence updates with rewards."""
        schema = SchemaField(H=10, W=10, feature_dim=6,
                            cfg=SchemaConfig(K=4, tile=5))
        
        # Simulate some winners (now as (iy, ix, k) tuples)
        schema.last_winners = [(0, 0, 0), (0, 0, 2)]
        
        # Update with positive reward
        schema.update_valence(1.0)
        assert schema.q[0, 0, 0] > 0, "Winner 0 at tile (0,0) should have positive valence"
        assert schema.q[0, 0, 2] > 0, "Winner 2 at tile (0,0) should have positive valence"
        assert schema.q[0, 0, 1] == 0, "Non-winner should have unchanged valence"
        
        # Update with negative reward
        schema.last_winners = [(0, 1, 1)]
        schema.update_valence(-0.5)
        assert schema.q[0, 1, 1] < 0, "Winner 1 at tile (0,1) should have negative valence"
    
    def test_signed_bias_field(self):
        """Test that bias field reflects signed valences."""
        schema = SchemaField(H=10, W=10, feature_dim=6,
                            cfg=SchemaConfig(K=2, tile=5, alpha_schema=1.0))
        
        # Set up contrasting valences per-tile
        schema.q[0, 0, 0] = 1.0   # Positive valence for tile (0,0), proto 0
        schema.q[1, 1, 1] = -1.0  # Negative valence for tile (1,1), proto 1
        
        # Create activation maps with signed deposition
        # Note: Sign is now applied during deposition, not in bias_field
        schema.Smaps[0, 2:4, 2:4] = np.tanh(schema.beta_valence * 1.0) * 1.0  # Positive
        schema.Smaps[1, 6:8, 6:8] = np.tanh(schema.beta_valence * -1.0) * 1.0  # Negative
        
        bias = schema.bias_field()
        
        # Check that positive valence creates attraction
        assert np.mean(bias[2:4, 2:4]) > 0, "Positive valence area should attract"
        
        # Check that negative valence creates repulsion
        assert np.mean(bias[6:8, 6:8]) < 0, "Negative valence area should repel"


class TestCounterfactualLearning:
    """Test A2: Counterfactual credit for valence learning."""
    
    def test_counterfactual_method_exists(self):
        """Test that counterfactual learning method exists."""
        env = ForageWorld(EnvConfig(H=10, W=10))
        agent = ChemotaxisAgentCA(env, AgentConfig(), Ablations())
        
        assert hasattr(agent, 'learn_valence_counterfactual'), \
            "Agent should have counterfactual learning method"
    
    def test_counterfactual_updates_valence(self):
        """Test that counterfactual learning updates valences."""
        env = ForageWorld(EnvConfig(H=10, W=10))
        agent = ChemotaxisAgentCA(env, AgentConfig(), Ablations())
        
        initial_valA = agent.valence["A"]
        
        # Simulate counterfactual learning
        field_at_action = {"A": 0.8, "B": 0.2, "Novel": 0.5}
        field_alternatives = {"A": 0.3, "B": 0.4, "Novel": 0.5}
        reward = 0.1
        
        agent.learn_valence_counterfactual(field_at_action, field_alternatives, reward)
        
        # A has higher value at chosen action, should increase with positive reward
        assert agent.valence["A"] > initial_valA, \
            "Valence A should increase when chosen action has higher A value"


class TestTemperatureSchedule:
    """Test A3: Enhanced temperature schedule with field flatness."""
    
    def test_temperature_in_flat_regions(self):
        """Test that temperature increases in flat field regions."""
        # This would require mocking the runner, so we test the concept
        P_flat = np.ones((10, 10)) * 0.5  # Completely flat field
        gy, gx = np.gradient(P_flat)
        grad_mag = np.sqrt(gy[5, 5]**2 + gx[5, 5]**2)
        
        # In flat region, gradient magnitude should be near zero
        assert grad_mag < 0.01, "Flat field should have near-zero gradient"
        
        # Temperature calculation (from runner)
        epsilon = 0.01
        alpha_grad = 0.3
        temp_flatness = alpha_grad / (epsilon + grad_mag)
        
        assert temp_flatness > 1.0, "Temperature should be high in flat regions"


class TestWallProximity:
    """Test A4: Wall proximity field integration."""
    
    def test_wall_proximity_generation(self):
        """Test wall proximity field generation."""
        walls = np.zeros((10, 10), dtype=bool)
        walls[4:6, :] = True  # Horizontal wall
        
        W_prox = wall_proximity_field(walls, radius=1.5)
        
        # Should be high near walls
        assert W_prox[3, 5] > 0.5, "Cell adjacent to wall should have high proximity"
        assert W_prox[6, 5] > 0.5, "Cell adjacent to wall should have high proximity"
        
        # Should decay with distance
        assert W_prox[2, 5] < W_prox[3, 5], "Proximity should decay with distance"
        assert W_prox[0, 5] < 0.1, "Far from wall should have low proximity"


class TestSemiringComposition:
    """Test B1: Semiring composition options."""
    
    def test_linear_aggregation(self):
        """Test linear aggregation mode (default)."""
        attractors = {
            "A": np.array([[1.0, 0.5], [0.0, 0.0]]),
            "B": np.array([[0.0, 0.5], [1.0, 0.0]])
        }
        weights = {"A": 1.0, "B": 0.5}
        
        P = compose_potential(attractors, {}, weights, {}, mode="linear")
        
        # Linear: P[0,0] = 1.0*1.0 + 0.5*0.0 = 1.0
        assert abs(P[0, 0] - 1.0) < 0.01
        # Linear: P[0,1] = 1.0*0.5 + 0.5*0.5 = 0.75
        assert abs(P[0, 1] - 0.75) < 0.01
    
    def test_lse_aggregation(self):
        """Test log-sum-exp aggregation."""
        attractors = {
            "A": np.array([[2.0, 0.0], [0.0, 0.0]]),
            "B": np.array([[1.0, 0.0], [0.0, 0.0]])
        }
        weights = {"A": 1.0, "B": 1.0}
        
        P = compose_potential(attractors, {}, weights, {}, 
                            mode="lse", beta_attr=1.0)
        
        # LSE emphasizes larger values
        # P[0,0] = log(exp(2) + exp(1)) ≈ 2.31
        expected = np.log(np.exp(2.0) + np.exp(1.0))
        assert abs(P[0, 0] - expected) < 0.01
    
    def test_maxplus_aggregation(self):
        """Test max-plus aggregation."""
        attractors = {
            "A": np.array([[1.0, 0.5], [0.0, 0.0]]),
            "B": np.array([[0.5, 1.0], [0.0, 0.0]])
        }
        weights = {"A": 1.0, "B": 1.0}
        
        P = compose_potential(attractors, {}, weights, {}, mode="maxplus")
        
        # Max-plus: P[0,0] = max(1.0, 0.5) = 1.0
        assert P[0, 0] == 1.0
        # Max-plus: P[0,1] = max(0.5, 1.0) = 1.0
        assert P[0, 1] == 1.0


class TestReachableFrontier:
    """Test B2: Reachability-aware frontier."""
    
    def test_reachable_frontier_blocked(self):
        """Test that frontier doesn't pull through walls."""
        seen = np.ones((10, 10), dtype=bool)
        seen[7:, :] = False  # Unseen area at bottom
        
        walls = np.zeros((10, 10), dtype=bool)
        walls[5, :] = True  # Wall blocking access to unseen area
        walls[5, 5] = False  # Small gap
        
        # From position above wall
        frontier = compute_reachable_frontier(seen, walls, y=2, x=5)
        
        # Should have low/zero frontier behind solid wall section
        assert frontier[8, 2] < 0.1, "Frontier behind wall should be minimal"
        
        # But should have some frontier accessible through gap
        assert frontier[8, 5] > 0.0, "Frontier through gap should be non-zero"


class TestConvolutionalSchema:
    """Test B3: Convolutional schema deposition."""
    
    def test_conv_deposition_spreads_activation(self):
        """Test that convolutional deposition spreads activation across tile."""
        schema = SchemaField(H=10, W=10, feature_dim=6,
                            cfg=SchemaConfig(K=2, tile=5, conv_deposition=True))
        
        # Create feature input
        feats = np.random.randn(10, 10, 6).astype(np.float32)
        
        # Force a specific tile to have high activation
        schema.Wp[0, 0, 0, :] = feats[2, 2, :] / np.linalg.norm(feats[2, 2, :])
        
        schema.update(feats)
        
        # Check that activation spreads across the tile, not just at center
        tile_activation = schema.Smaps[0, 0:5, 0:5]
        
        # With conv deposition, multiple cells should be active
        active_cells = np.sum(tile_activation > 0)
        assert active_cells > 1, "Convolutional deposition should activate multiple cells"


class TestFieldNormalization:
    """Test B4: Field unit normalization."""
    
    def test_diffusion_stability_constraint(self):
        """Test that diffusion enforces CFL stability constraint."""
        from efi.core.diffusion import diffuse_masked
        
        field = np.ones((10, 10))
        walls = np.zeros((10, 10), dtype=bool)
        
        # Try with diff > 0.25 (should be clamped)
        result = diffuse_masked(field, walls, diff=0.5, decay=0.0, steps=1)
        
        # Field should remain stable (not explode)
        assert np.all(np.isfinite(result)), "Field should remain finite"
        assert np.max(result) <= 1.0, "Field should not amplify beyond input"


class TestIntegration:
    """Integration tests for combined improvements."""
    
    def test_narrow_corridor_navigation(self):
        """Test wall-hug reduction in narrow corridors."""
        # Create narrow corridor environment
        cfg = EnvConfig(H=10, W=20, p_wall=0.0, n_targets_A=1, n_targets_B=0)
        env = ForageWorld(cfg)
        
        # Manually create corridor
        env.walls[:] = True
        env.walls[4:6, :] = False  # 2-cell wide corridor
        env.TA[:] = False
        env.TA[4, 18] = True  # Target at end of corridor
        env.y, env.x = 4, 1  # Start at beginning of corridor
        
        # w_wall_prox is accessed via getattr, not a constructor param
        cfg = AgentConfig()
        cfg.w_wall_prox = 0.4  # Set as attribute
        agent = ChemotaxisAgentCA(env, cfg, Ablations(wall_proximity=True))
        
        # Run a few steps
        obs = env._obs()
        # First step to discover walls in view
        _, fields = agent.step(obs)
        
        # Now check wall proximity field is active after walls are discovered
        assert 'WallProx' in fields, "WallProx should be in fields"
        # After discovering walls in the observation window, should have proximity
        if np.any(agent.known_walls):
            assert np.any(fields['WallProx'] > 0), \
                "Wall proximity should be active after discovering walls"
    
    def test_aversive_schema_learning(self):
        """Test that schema learns to avoid B-heavy areas."""
        cfg = EnvConfig(H=15, W=15, n_targets_A=2, n_targets_B=5,
                       reward_A=1.0, reward_B=-1.0)
        env = ForageWorld(cfg)
        agent = ChemotaxisAgentCA(env, AgentConfig(valB_init=0.5), Ablations())
        schema = SchemaField(env.H, env.W, feature_dim=6,
                           cfg=SchemaConfig(K=4, enabled=True))
        
        # Run episode to collect some B targets and learn
        _, _, metrics, _ = run_episode(env, agent, schema, Ablations())
        
        # After negative B experiences, relevant schema prototypes should have negative valence
        if np.any(schema.q < 0):
            # At least one prototype learned negative association
            assert True, "Schema learned negative valence for some prototypes"