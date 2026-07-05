"""Tests for ChemotaxisAgentCA."""

import numpy as np
import pytest

from efi.configs import EnvConfig, AgentConfig, Ablations
from efi.envs import ForageWorld
from efi.agents import ChemotaxisAgentCA
from efi.evaluation import run_episode


class TestChemotaxisAgent:
    """Test ChemotaxisAgentCA."""
    
    def test_agent_creation(self):
        """Test agent initialization."""
        env_cfg = EnvConfig(H=10, W=10, seed=42)
        env = ForageWorld(env_cfg)
        
        agent_cfg = AgentConfig(seed=42)
        ablate = Ablations()
        
        agent = ChemotaxisAgentCA(env, agent_cfg, ablate)
        
        assert agent.H == env.H
        assert agent.W == env.W
        assert agent.win == env.win
    
    def test_agent_reset(self):
        """Test agent reset."""
        env_cfg = EnvConfig(H=10, W=10, seed=42)
        env = ForageWorld(env_cfg)
        
        agent_cfg = AgentConfig(seed=42)
        ablate = Ablations()
        
        agent = ChemotaxisAgentCA(env, agent_cfg, ablate)
        agent.reset()
        
        # Check fields are initialized
        assert agent.GA.shape == (env.H, env.W)
        assert agent.GB.shape == (env.H, env.W)
        assert agent.V.shape == (env.H, env.W)
        assert agent.Nv.shape == (env.H, env.W)
        
        # Check fields are zero
        assert np.all(agent.GA == 0)
        assert np.all(agent.GB == 0)
        assert np.all(agent.V == 0)
        assert np.all(agent.Nv == 0)
    
    def test_agent_step(self):
        """Test agent step."""
        env_cfg = EnvConfig(H=10, W=10, seed=42)
        env = ForageWorld(env_cfg)
        obs = env.reset()
        
        agent_cfg = AgentConfig(seed=42)
        ablate = Ablations()
        
        agent = ChemotaxisAgentCA(env, agent_cfg, ablate)
        agent.reset()
        
        _, fields = agent.step(obs)
        
        # Check fields are returned
        assert "GA" in fields
        assert "GB" in fields
        assert "Vtrail" in fields
        assert "Novel" in fields
        assert "known_walls" in fields
        
        # Check field shapes
        assert fields["GA"].shape == (env.H, env.W)
        assert fields["GB"].shape == (env.H, env.W)
    
    def test_scent_seeding(self):
        """Test scent field seeding from observation."""
        env_cfg = EnvConfig(H=10, W=10, seed=42, n_targets_A=1, n_targets_B=0)
        env = ForageWorld(env_cfg)
        obs = env.reset()
        
        agent_cfg = AgentConfig(seed=42, seed_strength=1.0)
        ablate = Ablations()
        
        agent = ChemotaxisAgentCA(env, agent_cfg, ablate)
        agent.reset()
        
        # Place target A near agent
        target_y, target_x = env.y + 1, env.x
        if 0 <= target_y < env.H and 0 <= target_x < env.W:
            env.TA[:] = False
            env.TA[target_y, target_x] = True
            
            # Get new observation with target visible
            obs = env._obs()
            
            # Step agent
            _, fields = agent.step(obs)
            
            # Check that GA field has non-zero values
            assert np.any(fields["GA"] > 0)
    
    def test_ablations(self):
        """Test ablation flags."""
        env_cfg = EnvConfig(H=10, W=10, seed=42)
        env = ForageWorld(env_cfg)
        obs = env.reset()
        
        agent_cfg = AgentConfig(seed=42)
        
        # Test with trail disabled
        ablate = Ablations(trail=0, novelty=1, corner=1, schema=1)
        agent = ChemotaxisAgentCA(env, agent_cfg, ablate)
        agent.reset()
        
        _, fields = agent.step(obs)
        assert np.all(fields["Vtrail"] == 0)
        
        # Test with novelty disabled
        ablate = Ablations(trail=1, novelty=0, corner=1, schema=1)
        agent = ChemotaxisAgentCA(env, agent_cfg, ablate)
        agent.reset()
        
        _, fields = agent.step(obs)
        assert np.all(fields["Novel"] == 0)
    
    def test_stuck_detection(self):
        """stuck_count is owned by the runner: it increments after each
        failed move and resets on a successful one (see run_episode)."""
        env_cfg = EnvConfig(H=5, W=5, max_steps=6, seed=0)
        env = ForageWorld(env_cfg)

        # Box the agent in so that every move bumps a wall.
        original_reset = env.reset

        def boxed_reset():
            original_reset()
            env.walls[:] = True
            env.walls[2, 2] = False
            env.TA[:] = False
            env.TB[:] = False
            env.TA[0, 0] = True  # unreachable target keeps the episode alive
            env.y, env.x = 2, 2
            return env._obs()

        env.reset = boxed_reset

        agent_cfg = AgentConfig(seed=0, anti_stuck_after=3)
        ablate = Ablations()
        agent = ChemotaxisAgentCA(env, agent_cfg, ablate)

        run_episode(env, agent, None, ablate)

        # Every step bumped, so the runner incremented the counter each time.
        assert agent.stuck_count == env_cfg.max_steps