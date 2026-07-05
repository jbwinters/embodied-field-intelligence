"""Tests for ForageWorld environment."""

import numpy as np
import pytest

from efi.configs import EnvConfig
from efi.envs import ForageWorld


class TestForageWorld:
    """Test ForageWorld environment."""
    
    def test_env_creation(self):
        """Test environment initialization."""
        cfg = EnvConfig(H=10, W=10, seed=42)
        env = ForageWorld(cfg)
        
        assert env.H == 10
        assert env.W == 10
        assert env.win == cfg.win
        assert env.max_steps == cfg.max_steps
    
    def test_reset(self):
        """Test environment reset."""
        cfg = EnvConfig(H=10, W=10, seed=42)
        env = ForageWorld(cfg)
        
        obs = env.reset()
        
        # Check observation shape
        expected_shape = 4 * cfg.win * cfg.win
        assert obs.shape == (expected_shape,)
        
        # Check observation values are in [0, 1]
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)
        
        # Check agent spawned
        assert 0 <= env.y < env.H
        assert 0 <= env.x < env.W
        
        # Check targets exist
        assert np.sum(env.TA) == cfg.n_targets_A
        assert np.sum(env.TB) == cfg.n_targets_B
    
    def test_step(self):
        """Test environment step."""
        cfg = EnvConfig(H=10, W=10, seed=42)
        env = ForageWorld(cfg)
        env.reset()
        
        initial_y, initial_x = env.y, env.x
        
        # Try to move up (action 0)
        obs, reward, done, info = env.step(0)
        
        # Check return types
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        # Check observation shape
        expected_shape = 4 * cfg.win * cfg.win
        assert obs.shape == (expected_shape,)
    
    def test_wall_collision(self):
        """Test wall collision penalty."""
        cfg = EnvConfig(H=10, W=10, seed=42, bump_pen=-0.1)
        env = ForageWorld(cfg)
        env.reset()
        
        # Place agent next to wall
        env.y, env.x = 0, 5
        
        # Try to move up into boundary
        obs, reward, done, info = env.step(0)
        
        # Should get step cost + bump penalty
        expected_reward = cfg.step_cost + cfg.bump_pen
        assert reward == expected_reward
        assert not info["moved"]
    
    def test_target_collection(self):
        """Targets are collected by moving onto them; pickup is checked at
        the cell the agent ends the step on."""
        cfg = EnvConfig(H=10, W=10, seed=42)
        env = ForageWorld(cfg)
        env.reset()

        # Deterministic setup: open arena, one A target directly above agent
        env.walls[:] = False
        env.TA[:] = False
        env.TB[:] = False
        env.y, env.x = 5, 5
        env.TA[4, 5] = True

        obs, reward, done, info = env.step(0)  # move up onto the target

        assert info["picked"] == "A"
        assert reward == pytest.approx(cfg.reward_A + cfg.step_cost)
        assert not env.TA[4, 5]
        assert done  # last remaining target was collected
    
    def test_episode_termination(self):
        """Test episode termination conditions."""
        cfg = EnvConfig(H=10, W=10, max_steps=5, seed=42)
        env = ForageWorld(cfg)
        env.reset()
        
        # Run until max steps
        done = False
        for _ in range(cfg.max_steps):
            obs, reward, done, info = env.step(0)
        
        # Should be done after max steps
        assert done
    
    def test_render_rgb(self):
        """Test RGB rendering."""
        cfg = EnvConfig(H=10, W=10, seed=42)
        env = ForageWorld(cfg)
        env.reset()
        
        rgb = env.render_rgb()
        
        # Check shape and type
        assert rgb.shape == (env.H, env.W, 3)
        assert rgb.dtype == np.uint8
        
        # Check values in valid range
        assert np.all(rgb >= 0)
        assert np.all(rgb <= 255)