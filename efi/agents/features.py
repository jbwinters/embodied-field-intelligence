"""Feature extraction for schema learning."""

import numpy as np


def build_features_for_schema(GA, GB, Novel, Vtrail, Hc, P_eff) -> np.ndarray:
    """
    Build local CA-native feature stack per cell for schema learning.
    
    Features include:
    - GA: Target A scent
    - GB: Target B scent
    - Novel: Novelty field
    - Vtrail: Visit trail
    - Hc: Corner hazard
    - |∇P_eff|: Gradient magnitude of effective potential
    
    Args:
        GA: Target A scent field
        GB: Target B scent field
        Novel: Novelty field
        Vtrail: Visit trail field
        Hc: Corner hazard field
        P_eff: Effective potential field
        
    Returns:
        Feature array of shape (H, W, 6)
    """
    # Compute gradient magnitude of potential field
    gy, gx = np.gradient(P_eff)
    gradmag = np.sqrt(gy*gy + gx*gx).astype(np.float32)
    
    # Stack features
    feats = np.stack([
        GA.astype(np.float32),
        GB.astype(np.float32),
        Novel.astype(np.float32),
        Vtrail.astype(np.float32),
        Hc.astype(np.float32),
        gradmag
    ], axis=-1)
    
    return feats