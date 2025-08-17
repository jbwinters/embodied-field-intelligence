"""Gymnasium wrapper for ForageWorld."""

from typing import Optional

import numpy as np

from ..configs import EnvConfig
from ..core import set_global_seed
from .forage_world import ForageWorld


class _OptionalGym:
    """Lazy Gymnasium import."""
    
    def __init__(self):
        try:
            import gymnasium as gym
            from gymnasium import spaces
            self.gym = gym
            self.spaces = spaces
        except Exception:
            self.gym = None
            self.spaces = None


GYM = _OptionalGym()


class CAForageGymEnv(GYM.gym.Env if GYM.gym else object):
    """
    Thin Gymnasium wrapper for ForageWorld to interop with RL stacks.
    
    Observation: Box(float32) of shape (4*win*win,)
    Action: Discrete(4)
    """
    
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    def __init__(self, env_cfg: EnvConfig):
        """
        Initialize Gym wrapper.
        
        Args:
            env_cfg: Environment configuration
        """
        if not GYM.gym:
            raise ImportError("gymnasium is not installed. pip install gymnasium")
            
        self.env = ForageWorld(env_cfg)
        self.observation_space = GYM.spaces.Box(
            low=0.0, high=1.0, 
            shape=(4*env_cfg.win*env_cfg.win,), 
            dtype=np.float32
        )
        self.action_space = GYM.spaces.Discrete(4)
        self.render_mode = "rgb_array"

    def reset(self, seed: Optional[int] = None, options=None):
        """Reset environment."""
        if seed is not None:
            set_global_seed(int(seed))
        obs = self.env.reset()
        info = {}
        return obs.astype(np.float32), info

    def step(self, action: int):
        """Execute action."""
        obs, reward, done, info = self.env.step(int(action))
        terminated = done
        truncated = False
        return obs.astype(np.float32), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        """Render environment."""
        return self.env.render_rgb()


def register_gym_env(env_cfg: Optional[EnvConfig] = None):
    """
    Register environment with Gymnasium.
    
    Args:
        env_cfg: Optional environment configuration
    """
    if not GYM.gym:
        print("gymnasium is not installed. Skipping registration.")
        return
        
    from gymnasium.envs.registration import register
    
    if env_cfg is None:
        env_cfg = EnvConfig()
        
    def make_env():
        return CAForageGymEnv(env_cfg)
        
    try:
        register(id="CAForage-v0", entry_point=lambda: make_env())
        print("Gymnasium env registered as 'CAForage-v0'")
    except Exception as e:
        print(f"Registration may already exist or failed: {e}")