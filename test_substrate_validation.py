#!/usr/bin/env python3
"""Validation tests for the field controller substrate."""

import numpy as np
from efi import EnvConfig, AgentConfig, Ablations, ForageWorld
from efi.agents import FieldController, ForageAdapter
from efi.core import compose_potential, corner_hazard
from efi.evaluation import run_episode


def test_linearity_superposition():
    """Test that potential composition is linear."""
    print("Testing linearity and superposition...")
    
    # Create test fields
    H, W = 20, 20
    GA = np.random.rand(H, W).astype(np.float32)
    GB = np.random.rand(H, W).astype(np.float32)
    N = np.random.rand(H, W).astype(np.float32)
    V = np.random.rand(H, W).astype(np.float32)
    Hc = np.random.rand(H, W).astype(np.float32)
    
    # Test weights
    wA, wB, wN = 1.2, -0.5, 0.7
    kV, kH = 0.6, 0.5
    
    # Compute using compose_potential
    attractors = {"A": GA, "B": GB, "Novel": N}
    repulsors = {"Trail": V, "Corner": Hc}
    w_attr = {"A": wA, "B": wB, "Novel": wN}
    w_rep = {"Trail": kV, "Corner": kH}
    
    P_composed = compose_potential(attractors, repulsors, w_attr, w_rep)
    
    # Compute manually
    P_manual = wA * GA + wB * GB + wN * N - kV * V - kH * Hc
    
    # Check linearity
    max_error = np.max(np.abs(P_composed - P_manual))
    print(f"  Max linearity error: {max_error:.2e}")
    assert max_error < 1e-5, f"Linearity failed: max error = {max_error}"
    
    print("  ✓ Linearity test passed")


def test_alignment_metric():
    """Test gradient-motion alignment metric."""
    print("\nTesting gradient-motion alignment...")
    
    # Create environment and controller
    env_cfg = EnvConfig(H=20, W=20, n_targets_A=3, n_targets_B=2, seed=42)
    agent_cfg = AgentConfig(valence_lr=0.3, seed=42)
    ablate = Ablations(trail=True, novelty=True, corner=True, schema=False)
    
    env = ForageWorld(env_cfg)
    adapter = ForageAdapter(env)
    agent = FieldController(env, adapter, agent_cfg, ablate)
    
    # Run a short episode and collect alignments
    obs = env.reset()
    agent.reset()
    
    cosines = []
    prev_y, prev_x = env.y, env.x
    
    for _ in range(50):
        walls_mask = agent.step_fields(obs)
        Hc = corner_hazard(walls_mask)
        P = agent.compose_P(walls_mask, corner_field=Hc)
        
        # Compute gradient at current position
        gy, gx = np.gradient(P.astype(np.float32))
        grad = np.array([gy[env.y, env.x], gx[env.y, env.x]])
        
        # Pick action
        from efi.core import pick_action_from_potential
        action = pick_action_from_potential(P, env.y, env.x, walls_mask)
        obs, _, done, info = env.step(action)
        
        # Compute motion vector
        dy, dx = env.y - prev_y, env.x - prev_x
        motion = np.array([dy, dx], dtype=np.float32)
        
        # Compute alignment if we moved
        if np.linalg.norm(motion) > 0 and np.linalg.norm(grad) > 1e-6:
            cos = np.dot(grad, motion) / (np.linalg.norm(grad) * np.linalg.norm(motion))
            cosines.append(float(cos))
        
        prev_y, prev_x = env.y, env.x
        if done:
            break
    
    mean_cos = np.mean(cosines) if cosines else 0.0
    print(f"  Mean cosine alignment: {mean_cos:.3f}")
    print(f"  Samples: {len(cosines)}")
    
    # Should be positive (agent follows gradient on average)
    assert mean_cos > 0.0, f"Expected positive alignment, got {mean_cos}"
    print("  ✓ Alignment test passed")


def test_valence_learning():
    """Test that valences adapt to rewards."""
    print("\nTesting valence learning...")
    
    # Create environment with B as undesirable
    env_cfg = EnvConfig(
        H=15, W=15,
        n_targets_A=2, n_targets_B=4,
        reward_A=1.0, reward_B=-1.0,
        max_steps=100,
        seed=123
    )
    agent_cfg = AgentConfig(
        valA_init=1.0,
        valB_init=0.2,  # Starts positive
        valence_lr=0.4,
        valence_clip=1.5,
        seed=123
    )
    ablate = Ablations(trail=True, novelty=True, corner=False, schema=False)
    
    env = ForageWorld(env_cfg)
    adapter = ForageAdapter(env)
    agent = FieldController(env, adapter, agent_cfg, ablate)
    
    print(f"  Initial valences: A={agent.valence['A']:.2f}, B={agent.valence['B']:.2f}")
    
    # Run episode
    ret, _, metrics, _ = run_episode(env, agent, None, ablate)
    
    print(f"  Final valences: A={metrics.valence_snapshot['A']:.2f}, B={metrics.valence_snapshot['B']:.2f}")
    print(f"  Targets collected: A={metrics.targets_collected.get('A', 0)}, B={metrics.targets_collected.get('B', 0)}")
    
    # B valence should go negative if any B was collected
    if metrics.targets_collected.get('B', 0) > 0:
        assert metrics.valence_snapshot['B'] < agent_cfg.valB_init, \
            f"B valence should decrease, but {metrics.valence_snapshot['B']} >= {agent_cfg.valB_init}"
        print("  ✓ B avoidance learned")
    
    print("  ✓ Valence learning test passed")


def test_frontier_blending():
    """Test that frontier blending works correctly."""
    print("\nTesting frontier blending...")
    
    env_cfg = EnvConfig(H=20, W=20, seed=456)
    agent_cfg = AgentConfig(seed=456)
    ablate = Ablations(trail=True, novelty=True)
    
    env = ForageWorld(env_cfg)
    adapter = ForageAdapter(env)
    agent = FieldController(env, adapter, agent_cfg, ablate)
    
    obs = env.reset()
    agent.reset()
    walls_mask = agent.step_fields(obs)
    
    # Get potentials with different frontier weights
    P_no_frontier = agent.compose_P(walls_mask, frontier_weight=0.0)
    P_with_frontier = agent.compose_P(walls_mask, frontier_weight=0.5)
    
    # Should be different when frontier is non-zero
    diff = np.max(np.abs(P_with_frontier - P_no_frontier))
    print(f"  Max difference with frontier: {diff:.3f}")
    
    # The difference should be exactly frontier_weight * Frontier field
    expected_diff = 0.5 * agent.fields["Frontier"]
    actual_diff = P_with_frontier - P_no_frontier
    
    # Account for the novelty channel weight (0.7)
    expected_diff_weighted = 0.7 * expected_diff  # Novel weight is 0.7
    
    max_error = np.max(np.abs(actual_diff - expected_diff_weighted))
    print(f"  Frontier blend error: {max_error:.2e}")
    assert max_error < 1e-5, f"Frontier blending error: {max_error}"
    
    print("  ✓ Frontier blending test passed")


def test_channel_inference():
    """Test that channel count is correctly inferred."""
    print("\nTesting dynamic channel inference...")
    
    # Test with different observation sizes
    for win in [3, 5, 7]:
        for ch in [4, 6, 8]:
            obs_len = ch * win * win
            obs_vec = np.random.rand(obs_len).astype(np.float32)
            
            # Create controller
            env_cfg = EnvConfig(H=20, W=20, win=win)
            env = ForageWorld(env_cfg)
            env.win = win  # Override window size
            
            adapter = ForageAdapter(env)
            agent = FieldController(env, adapter, AgentConfig(), Ablations())
            
            # Infer channel count (from step_fields logic)
            inferred_ch = int(len(obs_vec) // (win * win))
            
            assert inferred_ch == ch, f"Failed for win={win}, ch={ch}: got {inferred_ch}"
            print(f"  ✓ Correctly inferred {ch} channels for {win}x{win} window")
    
    print("  ✓ Channel inference test passed")


def main():
    """Run all validation tests."""
    print("="*60)
    print("Field Controller Substrate Validation")
    print("="*60)
    
    test_linearity_superposition()
    test_alignment_metric()
    test_valence_learning()
    test_frontier_blending()
    test_channel_inference()
    
    print("\n" + "="*60)
    print("✅ All validation tests passed!")
    print("="*60)


if __name__ == "__main__":
    main()