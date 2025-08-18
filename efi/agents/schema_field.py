"""Schema field learning with Oja/BCM and slowness."""

import math

import numpy as np

from ..configs import SchemaConfig
from ..core import diffuse_masked


class SchemaField:
    """
    Local prototypes per tile learned with Oja/BCM + slowness.
    
    Produces slow, diffused schema activation maps S_k that bias P_eff.
    
    Learning uses:
    - Oja normalization for stable Hebbian learning
    - BCM sliding threshold for competition
    - Slowness penalty to encourage stable representations
    """
    
    def __init__(self, H: int, W: int, feature_dim: int, cfg: SchemaConfig):
        """
        Initialize schema field.
        
        Args:
            H, W: Grid dimensions
            feature_dim: Dimensionality of input features
            cfg: Schema configuration
        """
        self.cfg = cfg
        self.H, self.W = H, W
        self.d = feature_dim
        self.rng = np.random.RandomState(cfg.seed)
        
        # Tile grid dimensions
        self.th = cfg.tile
        self.ny = math.ceil(H / self.th)
        self.nx = math.ceil(W / self.th)
        
        # Prototypes: (ny, nx, K, d)
        self.Wp = self.rng.randn(self.ny, self.nx, cfg.K, self.d).astype(np.float32) * 0.05
        
        # Running postsynaptic threshold theta (BCM): (ny, nx, K)
        self.theta = np.zeros((self.ny, self.nx, cfg.K), dtype=np.float32)
        
        # Previous activation (for slowness): (ny, nx, K)
        self.y_prev = np.zeros((self.ny, self.nx, cfg.K), dtype=np.float32)
        
        # Schema activation maps (global)
        self.Smaps = np.zeros((cfg.K, H, W), dtype=np.float32)
        
        # Reward-correlated valence per tile, per prototype (for signed bias)
        self.q = np.zeros((self.ny, self.nx, cfg.K), dtype=np.float32)
        self.beta_valence = getattr(cfg, 'beta_valence', 2.0)  # Temperature for tanh
        self.rho_valence = getattr(cfg, 'rho_valence', 0.03)  # Learning rate for valence

    def _pool_tile_feature(self, feats: np.ndarray, iy: int, ix: int) -> np.ndarray:
        """
        Pool features from a tile region.
        
        Args:
            feats: Feature array (H, W, d)
            iy, ix: Tile indices
            
        Returns:
            Mean feature vector for tile
        """
        y0, x0 = iy * self.th, ix * self.th
        y1, x1 = min(y0 + self.th, self.H), min(x0 + self.th, self.W)
        tile = feats[y0:y1, x0:x1, :]
        
        if tile.size == 0:
            return np.zeros((self.d,), dtype=np.float32)
            
        return tile.mean(axis=(0,1)).astype(np.float32)

    def update(self, feats: np.ndarray):
        """
        Update prototypes and schema maps.
        
        Args:
            feats: (H, W, d) features built from fast fields each step
        """
        if not self.cfg.enabled:
            self.Smaps[:] = 0.0
            return

        ny, nx, K, d = self.ny, self.nx, self.cfg.K, self.d
        S_tmp = np.zeros((K, self.H, self.W), dtype=np.float32)
        self.last_winners = []  # Track (iy, ix, k) for reward updates

        for iy in range(ny):
            for ix in range(nx):
                x_tile = self._pool_tile_feature(feats, iy, ix)  # (d,)
                if not np.any(x_tile):
                    continue

                w = self.Wp[iy, ix]      # (K,d)
                y = w @ x_tile            # (K,)
                
                # Soft competition: choose top comp_k
                winners = np.argsort(y)[::-1][:max(1, self.cfg.comp_k)]
                for k in winners:
                    self.last_winners.append((iy, ix, int(k)))  # Track for reward update
                
                # BCM sliding threshold
                self.theta[iy, ix, :] = (
                    (1.0 - self.cfg.bcm_tau) * self.theta[iy, ix, :] + 
                    self.cfg.bcm_tau * (y * y)
                )

                # Oja/BCM + slowness on winners only
                for k in winners:
                    yk = y[k]
                    thk = self.theta[iy, ix, k]
                    
                    # BCM gain: y*(y - theta)
                    g = yk * (yk - thk)
                    
                    # Oja normalization: Δw = η * y * (x - y w)
                    dw = self.cfg.eta * yk * (x_tile - yk * w[k])
                    
                    # Slowness: penalize rapid change
                    dw -= self.cfg.slowness * (yk - self.y_prev[iy, ix, k]) * w[k] * 0.1
                    
                    w[k] += dw
                    
                    # Normalize to avoid blow-up
                    nrm = np.linalg.norm(w[k]) + 1e-6
                    w[k] /= nrm

                self.Wp[iy, ix] = w
                self.y_prev[iy, ix] = y

                # Convolutional deposition across the entire tile
                if getattr(self.cfg, 'conv_deposition', True):
                    # Deposit activation across the whole tile region
                    y0, x0 = iy * self.th, ix * self.th
                    y1, x1 = min(y0 + self.th, self.H), min(x0 + self.th, self.W)
                    
                    for k in range(K):
                        activation = max(0.0, y[k])
                        if activation > 0:
                            # Apply signed valence
                            signed = np.tanh(self.beta_valence * self.q[iy, ix, k])
                            # Apply a simple box kernel (uniform within tile)
                            # Could also use Gaussian or other kernels
                            tile_size = (y1 - y0) * (x1 - x0)
                            if tile_size > 0:
                                # Normalize by tile size to maintain overall magnitude
                                S_tmp[k, y0:y1, x0:x1] += signed * activation / np.sqrt(tile_size)
                else:
                    # Legacy: deposit at tile center only
                    cy = min(self.H-1, iy*self.th + self.th//2)
                    cx = min(self.W-1, ix*self.th + self.th//2)
                    for k in range(K):
                        activation = max(0.0, y[k])
                        signed = np.tanh(self.beta_valence * self.q[iy, ix, k])
                        S_tmp[k, cy, cx] += signed * activation

        # Diffuse to produce smooth schema maps
        for k in range(self.cfg.K):
            self.Smaps[k] = diffuse_masked(
                S_tmp[k], 
                np.zeros_like(S_tmp[k], dtype=bool),
                diff=self.cfg.diff, 
                decay=self.cfg.decay, 
                steps=self.cfg.steps
            )

    def update_valence(self, reward: float):
        """
        Update valence weights based on experienced reward.
        
        Args:
            reward: The reward signal to associate with recent winners
        """
        if not self.cfg.enabled or not hasattr(self, 'last_winners'):
            return
            
        # Update valence for recent winning prototypes
        for (iy, ix, k) in set(self.last_winners):
            self.q[iy, ix, k] = (1.0 - self.rho_valence) * self.q[iy, ix, k] + self.rho_valence * reward

    def bias_field(self) -> np.ndarray:
        """
        Generate bias field from schema activations.
        
        Note: Sign is already applied during deposition, so we just sum.
        
        Returns:
            Bias field to add to P_eff (can be positive or negative)
        """
        if not self.cfg.enabled:
            return np.zeros((self.H, self.W), dtype=np.float32)
            
        # Smaps already contain signed deposition; just sum them
        Ssum = np.sum(self.Smaps, axis=0)
        return self.cfg.alpha_schema * Ssum.astype(np.float32)