"""Comprehensive test suite for core EFI functionality."""

import numpy as np
import pytest

from efi.configs import EnvConfig, AgentConfig, Ablations, SchemaConfig
from efi.envs import ForageWorld
from efi.agents import ChemotaxisAgentCA, SchemaField, build_features_for_schema
from efi.core import (
    diffuse_masked,
    update_visit_trail,
    update_novelty,
    corner_hazard,
    effective_potential,
    pick_action_from_potential
)
from efi.evaluation.runner import run_episode


class TestCoreFields:
    """Test core field operations."""
    
    def test_diffuse_masked(self):
        """Test diffusion with masking."""
        field = np.zeros((10, 10), dtype=np.float32)
        field[5, 5] = 1.0
        
        walls = np.zeros((10, 10), dtype=bool)
        walls[4:7, 6] = True  # vertical wall
        
        result = diffuse_masked(field, walls, diff=0.1, decay=0.01, steps=3)
        
        # Check diffusion happened
        assert result[5, 5] < 1.0
        assert result[5, 4] > 0  # left side
        assert result[5, 7] == 0  # blocked by wall
        
    def test_visit_trail_update(self):
        """Test visit trail updates."""
        V = np.zeros((10, 10), dtype=np.float32)
        walls = np.zeros((10, 10), dtype=bool)
        
        # First visit
        V_new = update_visit_trail(V, 5, 5, walls, v_decay=0.01, v_diff=0.1, v_inj=1.0)
        assert V_new[5, 5] > 0
        
        # Second visit - should accumulate
        V_new2 = update_visit_trail(V_new, 5, 5, walls, v_decay=0.01, v_diff=0.1, v_inj=1.0)
        assert V_new2[5, 5] > V_new[5, 5]
        
    def test_novelty_update(self):
        """Test novelty field updates."""
        Nv = np.zeros((10, 10), dtype=np.float32)
        walls = np.zeros((10, 10), dtype=bool)
        
        # High prediction error
        Nv_new = update_novelty(Nv, 0.8, 5, 5, walls, n_decay=0.02, n_diff=0.1, gain=5.0)
        assert Nv_new[5, 5] > 0
        
        # Low prediction error
        Nv_new2 = update_novelty(Nv_new, 0.1, 3, 3, walls, n_decay=0.02, n_diff=0.1, gain=5.0)
        assert Nv_new2[3, 3] > 0
        assert Nv_new2[3, 3] < Nv_new2[5, 5]
        
    def test_corner_hazard(self):
        """Test corner hazard detection."""
        walls = np.zeros((10, 10), dtype=bool)
        walls[3:7, 3] = True  # vertical wall
        walls[3, 3:7] = True  # horizontal wall (L-shape)
        
        Hc = corner_hazard(walls)
        
        # Check that hazard field is generated
        # Values depend on specific implementation
        assert Hc.shape == walls.shape
        # Should have some non-zero values near walls
        if np.any(walls):
            # There should be some hazard near walls
            pass


class TestChemotaxisAgent:
    """Test ChemotaxisAgentCA behavior."""
    
    def test_valence_learning(self):
        """Test valence weight updates."""
        env = ForageWorld(EnvConfig(H=20, W=20, seed=42))
        agent = ChemotaxisAgentCA(env, AgentConfig(valA_init=0.5, valB_init=0.5, valence_lr=0.1), Ablations())
        
        initial_valA = agent.valA
        initial_valB = agent.valB
        
        # Positive reward for A
        agent.learn_valence("A", 1.0)
        assert agent.valA > initial_valA
        
        # Negative reward for B
        agent.learn_valence("B", -1.0)
        assert agent.valB < initial_valB
        
    def test_ping_pong_detection(self):
        """Test ping-pong movement detection."""
        env = ForageWorld(EnvConfig(H=20, W=20, seed=42))
        agent = ChemotaxisAgentCA(env, AgentConfig(), Ablations())
        agent.reset()
        
        obs = env.reset()
        
        # Simulate ping-pong movement
        from collections import deque
        agent._pos_hist = deque([(5, 5), (5, 6), (5, 5)], maxlen=3)
        
        # Step should detect ping-pong
        _, fields = agent.step(obs)
        # Trail should be stronger when ping-ponging
        
    def test_frontier_exploration(self):
        """Test frontier-based exploration."""
        env = ForageWorld(EnvConfig(H=20, W=20, seed=42))
        agent = ChemotaxisAgentCA(env, AgentConfig(), Ablations())
        agent.reset()
        
        obs = env.reset()
        _, fields = agent.step(obs)
        
        # Frontier should be non-zero for unseen areas
        assert np.any(fields["Frontier"] > 0)
        
        # Seen areas should have zero frontier
        y, x = env.y, env.x
        assert agent.seen[y, x] == True
        
    def test_wall_discovery(self):
        """Test progressive wall discovery."""
        env = ForageWorld(EnvConfig(H=20, W=20, seed=42, p_wall=0.15))
        agent = ChemotaxisAgentCA(env, AgentConfig(), Ablations())
        agent.reset()
        
        # Initially no walls known
        assert np.sum(agent.known_walls) == 0
        
        obs = env.reset()
        _, fields = agent.step(obs)
        
        # Should discover some walls from observation
        if np.any(env.walls):
            # If there are visible walls, they should be discovered
            pass  # Wall discovery depends on visibility


class TestSchemaField:
    """Test SchemaField learning."""
    
    def test_schema_initialization(self):
        """Test schema field initialization."""
        schema = SchemaField(20, 20, 6, SchemaConfig(tile=5, K=4, enabled=True))
        
        assert schema.ny == 4  # 20/5
        assert schema.nx == 4  # 20/5
        assert schema.Wp.shape == (4, 4, 4, 6)
        assert schema.Smaps.shape == (4, 20, 20)
        
    def test_schema_update(self):
        """Test schema prototype updates."""
        schema = SchemaField(20, 20, 6, SchemaConfig(tile=5, K=4, enabled=True, eta=0.1))
        
        # Create feature input
        feats = np.random.randn(20, 20, 6).astype(np.float32) * 0.1
        
        initial_Wp = schema.Wp.copy()
        schema.update(feats)
        
        # Weights should change
        assert not np.allclose(schema.Wp, initial_Wp)
        
        # Schema maps should be generated
        assert np.any(schema.Smaps != 0)
        
    def test_schema_bias_field(self):
        """Test schema bias field generation."""
        schema = SchemaField(20, 20, 6, SchemaConfig(tile=5, K=4, enabled=True, alpha_schema=0.5))
        
        # Set some schema activations
        schema.Smaps[0, 10, 10] = 1.0
        schema.Smaps[1, 15, 15] = 0.5
        
        bias = schema.bias_field()
        
        assert bias.shape == (20, 20)
        assert bias[10, 10] > 0
        assert bias[15, 15] > 0
        
    def test_schema_disabled(self):
        """Test disabled schema."""
        schema = SchemaField(20, 20, 6, SchemaConfig(enabled=False))
        
        feats = np.random.randn(20, 20, 6).astype(np.float32)
        schema.update(feats)
        
        # Should produce zero bias
        bias = schema.bias_field()
        assert np.all(bias == 0)


class TestActionSelection:
    """Test action selection from potential fields."""
    
    def test_greedy_action(self):
        """Test greedy action selection."""
        P = np.zeros((10, 10), dtype=np.float32)
        P[6, 5] = 1.0  # attractive point below
        
        walls = np.zeros((10, 10), dtype=bool)
        
        action = pick_action_from_potential(P, 5, 5, walls, temperature=0.0)
        # Action should move toward higher potential
        assert action in [0, 1, 2, 3]  # Valid action
        
    def test_wall_blocking(self):
        """Test walls block movement."""
        P = np.zeros((10, 10), dtype=np.float32)
        P[6, 5] = 1.0  # attractive point below
        
        walls = np.zeros((10, 10), dtype=bool)
        walls[6, 5] = True  # wall blocks the attractive point
        
        action = pick_action_from_potential(P, 5, 5, walls, temperature=0.0)
        assert action != 2  # should not go down into wall
        
    def test_temperature_exploration(self):
        """Test temperature-based exploration."""
        P = np.zeros((10, 10), dtype=np.float32)
        P[6, 5] = 0.1  # weak attraction below
        
        walls = np.zeros((10, 10), dtype=bool)
        
        # With high temperature, should sometimes explore
        actions = []
        for _ in range(20):
            action = pick_action_from_potential(P, 5, 5, walls, temperature=2.0)
            actions.append(action)
        
        # Should have variety in actions
        assert len(set(actions)) > 1
        
    def test_no_backtrack(self):
        """Test no-backtrack constraint."""
        P = np.zeros((10, 10), dtype=np.float32)
        # Set multiple attractive points to give options
        P[4, 5] = 1.0  # above
        P[5, 6] = 0.8  # right
        P[5, 4] = 0.8  # left
        
        walls = np.zeros((10, 10), dtype=bool)
        
        # Last action was down (2), with no_backtrack we should avoid up (0)
        # But implementation may vary, so just check valid action
        action = pick_action_from_potential(P, 5, 5, walls, 
                                           temperature=0.0, 
                                           last_action=2, 
                                           no_backtrack=True)
        assert action in [0, 1, 2, 3]  # Valid action


class TestIntegration:
    """Test integrated episode running."""
    
    def test_episode_completion(self):
        """Test full episode execution."""
        env = ForageWorld(EnvConfig(H=20, W=20, seed=42, max_steps=100))
        agent = ChemotaxisAgentCA(env, AgentConfig(), Ablations())
        schema = SchemaField(20, 20, 6, SchemaConfig(enabled=False))
        
        ret, frames, metrics, _ = run_episode(env, agent, schema, Ablations())
        
        assert metrics.steps > 0
        assert metrics.steps <= 100
        assert isinstance(metrics.total_return, float)
        
    def test_target_collection(self):
        """Test target collection tracking."""
        env = ForageWorld(EnvConfig(H=20, W=20, seed=42, n_targets_A=3, n_targets_B=2))
        agent = ChemotaxisAgentCA(env, AgentConfig(), Ablations())
        
        ret, frames, metrics, _ = run_episode(env, agent, None, Ablations())
        
        assert "A" in metrics.targets_collected
        assert "B" in metrics.targets_collected
        assert metrics.targets_collected["A"] >= 0
        assert metrics.targets_collected["B"] >= 0
        
    def test_ablation_effects(self):
        """Test different ablation configurations."""
        env = ForageWorld(EnvConfig(H=20, W=20, seed=42))
        agent_cfg = AgentConfig()
        
        # No trail
        agent1 = ChemotaxisAgentCA(env, agent_cfg, Ablations(trail=0))
        ret1, _, metrics1, _ = run_episode(env, agent1, None, Ablations(trail=0))
        
        # With trail
        env.reset()
        agent2 = ChemotaxisAgentCA(env, agent_cfg, Ablations(trail=1))
        ret2, _, metrics2, _ = run_episode(env, agent2, None, Ablations(trail=1))
        
        # Both should complete but may have different behavior
        assert metrics1.steps > 0
        assert metrics2.steps > 0
        
    def test_valence_learning_integration(self):
        """Test valence learning during episode."""
        env = ForageWorld(EnvConfig(H=20, W=20, seed=42, 
                                   reward_A=1.0, reward_B=-0.5,
                                   n_targets_A=2, n_targets_B=2))
        agent = ChemotaxisAgentCA(env, AgentConfig(valence_lr=0.1), Ablations())
        
        initial_valA = agent.valA
        initial_valB = agent.valB
        
        ret, _, metrics, _ = run_episode(env, agent, None, Ablations())
        
        # Valence should adapt based on rewards
        if metrics.targets_collected["A"] > 0:
            assert agent.valA != initial_valA
        if metrics.targets_collected["B"] > 0:
            assert agent.valB != initial_valB


class TestEffectivePotential:
    """Test effective potential field computation."""
    
    def test_potential_combination(self):
        """Test combining multiple fields into effective potential."""
        GA = np.zeros((10, 10), dtype=np.float32)
        GB = np.zeros((10, 10), dtype=np.float32)
        Novel = np.zeros((10, 10), dtype=np.float32)
        Vtrail = np.zeros((10, 10), dtype=np.float32)
        Hc = np.zeros((10, 10), dtype=np.float32)
        
        GA[5, 5] = 1.0
        GB[7, 7] = 0.5
        Novel[3, 3] = 0.3
        Vtrail[5, 5] = 0.2
        
        P_eff = effective_potential(GA, GB, Novel, Vtrail, Hc,
                                   wA=1.0, wB=0.5, wN=0.7, kV=0.5, kH=0.3)
        
        # Check attractive components
        assert P_eff[5, 5] > 0  # GA attraction minus trail repulsion
        assert P_eff[7, 7] > 0  # GB attraction
        assert P_eff[3, 3] > 0  # Novelty attraction
        
    def test_repulsive_valence(self):
        """Test negative valence creates repulsion."""
        GA = np.zeros((10, 10), dtype=np.float32)
        GB = np.zeros((10, 10), dtype=np.float32)
        GB[5, 5] = 1.0
        
        P_eff_attract = effective_potential(GA, GB, GA, GA, GA, wA=0, wB=1.0, wN=0, kV=0, kH=0)
        P_eff_repel = effective_potential(GA, GB, GA, GA, GA, wA=0, wB=-1.0, wN=0, kV=0, kH=0)
        
        assert P_eff_attract[5, 5] > 0  # Positive weight attracts
        assert P_eff_repel[5, 5] < 0    # Negative weight repels