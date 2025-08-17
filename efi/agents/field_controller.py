"""General field-based controller using channel-agnostic potential composition."""

from typing import Dict, Optional, Tuple
from collections import deque
import numpy as np

from ..core import diffuse_masked, update_visit_trail, update_novelty
from ..core.potential import compose_potential
from .adapters import ControllerAdapter


class FieldController:
    """
    General field-based controller that composes arbitrary attractor/repulsor channels.
    
    This controller:
    - Maintains arbitrary named fields (attractors and repulsors)
    - Learns valences (weights) per channel through experience
    - Composes a single potential field from all influences
    - Provides action selection based on potential gradient
    
    The controller is environment-agnostic through the adapter interface.
    """
    
    def __init__(self, env, adapter: ControllerAdapter, cfg, ablate, seed=0):
        """
        Initialize field controller.
        
        Args:
            env: Environment instance
            adapter: Adapter for environment-specific mappings
            cfg: Configuration object
            ablate: Ablation flags
            seed: Random seed
        """
        self.env = env
        self.adapter = adapter
        self.cfg = cfg
        self.ablate = ablate
        self.rng = np.random.RandomState(seed)
        
        self.H, self.W = env.H, env.W
        self.win = getattr(env, "win", 1)
        
        # Initialize fast fields
        self.fields = {
            "A": np.zeros((self.H, self.W), dtype=np.float32),
            "B": np.zeros((self.H, self.W), dtype=np.float32),
            "Novel": np.zeros((self.H, self.W), dtype=np.float32),
            "Trail": np.zeros((self.H, self.W), dtype=np.float32),
            "Frontier": np.zeros((self.H, self.W), dtype=np.float32),
        }
        
        # Known walls for diffusion blocking (discovered over time)
        self.known_walls = np.zeros((self.H, self.W), dtype=bool)
        self.seen = np.zeros((self.H, self.W), dtype=bool)
        
        # Learnable valence table
        self.valence = {
            "A": float(cfg.valA_init),
            "B": float(cfg.valB_init),
            "Novel": 0.7,  # Can optionally learn slowly
        }
        
        # Backwards compatibility attributes
        self.valA = self.valence["A"]
        self.valB = self.valence["B"]
        
        # State tracking
        self.prev_P_here = 0.0
        self._prev_patch = None
        self._pos_hist = deque(maxlen=3)
        self._pos_hist.append((self.env.y, self.env.x))
        self.last_action = None
        self.stuck_count = 0
        
    def reset(self):
        """Reset controller state for new episode."""
        # Clear all fields
        for k in self.fields:
            self.fields[k][:] = 0.0
            
        self.known_walls[:] = False
        self.seen[:] = False
        
        # Mark initial visible area as seen
        y, x = self.env.y, self.env.x
        half = self.win // 2
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                gy, gx = y + dy, x + dx
                if 0 <= gy < self.H and 0 <= gx < self.W:
                    self.seen[gy, gx] = True
        
        # Reset state tracking
        self.prev_P_here = 0.0
        self._prev_patch = None
        self._pos_hist.clear()
        self._pos_hist.append((self.env.y, self.env.x))
        self.last_action = None
        self.stuck_count = 0
        
    def learn_valence(self, channel: str, reward: float):
        """
        Update valence weight for a channel based on reward.
        
        Args:
            channel: Name of the channel to update
            reward: Reward signal for learning
        """
        lr = float(self.cfg.valence_lr)
        clip = float(getattr(self.cfg, "valence_clip", 1.5))
        current = self.valence.get(channel, 0.0)
        self.valence[channel] = float(np.clip(current + lr * reward, -clip, clip))
        
        # Update backwards compatibility attributes
        if channel == "A":
            self.valA = self.valence["A"]
        elif channel == "B":
            self.valB = self.valence["B"]
        
    def step_fields(self, obs_vec: np.ndarray) -> np.ndarray:
        """
        Update all fields based on observation.
        
        Args:
            obs_vec: Flattened observation vector
            
        Returns:
            Walls mask for diffusion operations
        """
        y, x = self.env.y, self.env.x
        
        # 1) Get seeds from observation via adapter
        seeds = self.adapter.seed_from_obs(obs_vec, y, x)
        
        # Update known walls and mark area as seen
        ch = 4  # TODO: Get from adapter
        patch = obs_vec[:ch * self.win * self.win].reshape(ch, self.win, self.win)
        half = self.win // 2
        
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                gy, gx = y + dy, x + dx
                if 0 <= gy < self.H and 0 <= gx < self.W:
                    self.seen[gy, gx] = True
        
        for (gy, gx) in seeds.get("walls", []):
            self.known_walls[gy, gx] = True
            
        # Seed attractor fields
        for (gy, gx) in seeds.get("A", []):
            self.fields["A"][gy, gx] = max(self.fields["A"][gy, gx], self.cfg.seed_strength)
        for (gy, gx) in seeds.get("B", []):
            self.fields["B"][gy, gx] = max(self.fields["B"][gy, gx], self.cfg.seed_strength)
            
        walls_mask = self.known_walls.copy()
        
        # 2) Diffuse attractor channels
        for channel in ("A", "B"):
            self.fields[channel] = diffuse_masked(
                self.fields[channel], walls_mask,
                diff=self.cfg.scent_diff,
                decay=self.cfg.scent_decay,
                steps=self.cfg.scent_steps
            )
            
        # 3) Trail (repulsor) with ping-pong detection
        self._pos_hist.append((y, x))
        pingpong = (len(self._pos_hist) == 3 and
                   self._pos_hist[0] == self._pos_hist[2] and
                   self._pos_hist[0] != self._pos_hist[1])
        
        if self.ablate.trail:
            v_inj = self.cfg.v_inj * (2.5 if pingpong else 1.0)
            self.fields["Trail"] = update_visit_trail(
                self.fields["Trail"], y, x, walls_mask,
                v_decay=self.cfg.v_decay,
                v_diff=self.cfg.v_diff,
                v_inj=v_inj
            )
        else:
            self.fields["Trail"][:] = 0.0
            
        # 4) Novelty (attractor) based on prediction error
        P_local = float(self.fields["A"][y, x] + self.fields["B"][y, x])
        d_scent = abs(P_local - self.prev_P_here)
        self.prev_P_here = P_local
        
        # Observation change
        d_obs = 0.0
        if self._prev_patch is not None:
            d_obs = float(np.mean(np.abs(patch[:3] - self._prev_patch[:3])))
        self._prev_patch = patch.copy()
        
        pred_err = 0.5 * d_scent + 0.5 * d_obs
        
        if self.ablate.novelty:
            self.fields["Novel"] = update_novelty(
                self.fields["Novel"], pred_err, y, x, walls_mask,
                n_decay=0.018, n_diff=0.06, gain=6.0
            )
        else:
            self.fields["Novel"][:] = 0.0
            
        # 5) Frontier field (unseen areas)
        U = (~self.seen).astype(np.float32)
        U = U * (~walls_mask).astype(np.float32)  # Mask out walls
        U = diffuse_masked(U, walls_mask, diff=0.15, decay=0.01, steps=3)
        self.fields["Frontier"] = U
        
        return walls_mask
        
    def compose_P(self, walls_mask: np.ndarray,
                  corner_field: Optional[np.ndarray] = None,
                  schema_bias: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compose potential field from all influences.
        
        Args:
            walls_mask: Boolean mask of walls
            corner_field: Optional corner hazard field
            schema_bias: Optional schema bias field
            
        Returns:
            Composed potential field
        """
        # Prepare attractors and repulsors
        attractors = {
            "A": self.fields["A"],
            "B": self.fields["B"],
            "Novel": self.fields["Novel"],
        }
        
        repulsors = {
            "Trail": self.fields["Trail"],
        }
        
        if corner_field is not None:
            repulsors["Corner"] = corner_field
            
        # Get weights
        w_attr = {
            "A": self.valence.get("A", 1.0),
            "B": self.valence.get("B", 1.0),
            "Novel": self.valence.get("Novel", 0.7),
        }
        
        w_rep = {
            "Trail": 0.6,
            "Corner": 0.5 if corner_field is not None else 0.0,
        }
        
        # Compose potential
        return compose_potential(attractors, repulsors, w_attr, w_rep, bias=schema_bias)
        
    def step(self, obs_vec: np.ndarray) -> Tuple[int, Dict[str, np.ndarray]]:
        """
        Process observation and update internal fields.
        Compatible with ChemotaxisAgentCA interface for drop-in replacement.
        
        Args:
            obs_vec: Flattened observation vector
            
        Returns:
            Tuple of (action=-1, fields dictionary)
        """
        self.step_fields(obs_vec)
        return -1, self.get_fields()
    
    def get_fields(self) -> Dict[str, np.ndarray]:
        """
        Get all current fields for visualization/debugging.
        
        Returns:
            Dictionary of field copies
        """
        return {
            "GA": self.fields["A"].copy(),
            "GB": self.fields["B"].copy(),
            "Vtrail": self.fields["Trail"].copy(),
            "Novel": self.fields["Novel"].copy(),
            "Frontier": self.fields["Frontier"].copy(),
            "known_walls": self.known_walls.copy(),
        }