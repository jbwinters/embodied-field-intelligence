"""Adapter interfaces for mapping between environments and field controllers."""

from typing import Dict, List, Tuple
import numpy as np


class ControllerAdapter:
    """
    Abstract adapter interface for mapping environment observations to field deposits
    and action spaces.
    
    This allows the same field controller to work with different environments
    by abstracting environment-specific details.
    """
    
    def seed_from_obs(self, obs_vec: np.ndarray, y: int, x: int) -> Dict[str, List[Tuple[int, int]]]:
        """
        Convert observation into field seed locations.
        
        Args:
            obs_vec: Flattened observation vector from environment
            y, x: Current agent position
            
        Returns:
            Dictionary mapping channel names to lists of (y, x) seed positions
        """
        raise NotImplementedError
    
    def walls_mask(self) -> np.ndarray:
        """
        Get the current walls mask.
        
        Returns:
            Boolean array where True indicates wall positions
        """
        raise NotImplementedError
    
    def discrete(self) -> bool:
        """
        Whether this adapter is for a discrete action space.
        
        Returns:
            True for discrete (grid) environments, False for continuous
        """
        return True


class ForageAdapter(ControllerAdapter):
    """
    Adapter for ForageWorld environment.
    
    Maps patch observations to A/B/wall seeds and maintains wall knowledge.
    """
    
    def __init__(self, env):
        """
        Initialize adapter with ForageWorld environment.
        
        Args:
            env: ForageWorld environment instance
        """
        self.env = env
        self.H, self.W = env.H, env.W
        self.win = env.win
        
    def seed_from_obs(self, obs_vec: np.ndarray, y: int, x: int) -> Dict[str, List[Tuple[int, int]]]:
        """
        Extract A, B, and wall positions from observation patch.
        
        Args:
            obs_vec: Flattened observation vector [walls, A, B, agent, ...]
            y, x: Current agent position
            
        Returns:
            Dictionary with "A", "B", and "walls" seed locations
        """
        ch = 4  # Number of channels in observation
        patch = obs_vec[:ch * self.win * self.win].reshape(ch, self.win, self.win)
        
        walls_local = (patch[0] > 0.5)
        A_local = (patch[1] > 0.5)
        B_local = (patch[2] > 0.5)
        
        half = self.win // 2
        
        seeds = {"A": [], "B": [], "walls": []}
        
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                gy, gx = y + dy, x + dx
                py, px = dy + half, dx + half
                
                if 0 <= gy < self.H and 0 <= gx < self.W:
                    if walls_local[py, px]:
                        seeds["walls"].append((gy, gx))
                    if A_local[py, px]:
                        seeds["A"].append((gy, gx))
                    if B_local[py, px]:
                        seeds["B"].append((gy, gx))
        
        return seeds
    
    def walls_mask(self) -> np.ndarray:
        """
        Get the environment's wall mask.
        
        Returns:
            Boolean array of wall positions
        """
        return self.env.walls.copy()
    
    def discrete(self) -> bool:
        """
        ForageWorld uses discrete 4-directional actions.
        
        Returns:
            True (discrete action space)
        """
        return True