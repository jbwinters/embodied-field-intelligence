#!/usr/bin/env python3
"""Test script for the generalized field controller."""

import numpy as np
from dataclasses import dataclass

from efi.configs import AgentConfig, EnvConfig, Ablations
from efi.envs import ForageWorld
from efi.agents import ChemotaxisAgentCA, FieldController, ForageAdapter
from efi.evaluation import run_episode


def test_field_controller():
    """Test that the new FieldController works as a drop-in replacement."""
    
    print("Testing generalized field controller...")
    
    # Create environment
    env_cfg = EnvConfig(
        H=30, W=30,
        n_targets_A=2,
        n_targets_B=4,
        reward_A=1.0,
        reward_B=-0.5,  # B is undesirable
        seed=42
    )
    env = ForageWorld(env_cfg)
    
    # Create agent config
    agent_cfg = AgentConfig(
        valA_init=1.0,
        valB_init=0.1,  # Start slightly positive so agent discovers it's bad
        valence_lr=0.25,
        valence_clip=1.5,
        seed_strength=1.0,
        scent_diff=0.12,
        scent_decay=0.008,
        scent_steps=3,
        v_decay=0.012,
        v_diff=0.08,
        v_inj=1.0,
        seed=42
    )
    
    # Create ablations (all features on)
    ablate = Ablations(
        trail=True,
        novelty=True,
        corner=True,
        schema=False  # Keep schema off for simpler test
    )
    
    # Test 1: Original ChemotaxisAgentCA with dictionary valences
    print("\n1. Testing refactored ChemotaxisAgentCA...")
    agent1 = ChemotaxisAgentCA(env, agent_cfg, ablate)
    
    # Verify it has the new valence dictionary
    assert hasattr(agent1, 'valence'), "Agent should have valence dictionary"
    assert isinstance(agent1.valence, dict), "Valence should be a dictionary"
    assert 'A' in agent1.valence and 'B' in agent1.valence, "Should have A and B channels"
    print(f"   Initial valences: {agent1.valence}")
    
    # Run a short episode
    ep_return, frames, metrics1, episode_data = run_episode(env, agent1, None, ablate)
    print(f"   Episode return: {metrics1.total_return:.2f}")
    print(f"   Targets collected: {metrics1.targets_collected}")
    print(f"   Final valences: {metrics1.valence_snapshot}")
    print(f"   Mean gradient-motion alignment: {metrics1.mean_cosine:.3f}" if metrics1.mean_cosine else "   No cosine data")
    
    # Test 2: New FieldController
    print("\n2. Testing new FieldController...")
    env.reset()  # Reset environment
    adapter = ForageAdapter(env)
    agent2 = FieldController(env, adapter, agent_cfg, ablate)
    
    # Verify structure
    assert hasattr(agent2, 'valence'), "Controller should have valence dictionary"
    assert hasattr(agent2, 'fields'), "Controller should have fields dictionary"
    assert 'A' in agent2.fields and 'B' in agent2.fields, "Should have A and B fields"
    print(f"   Initial valences: {agent2.valence}")
    
    # Test field update
    obs = env.reset()
    walls_mask = agent2.step_fields(obs)
    print(f"   Fields updated successfully")
    print(f"   Known walls: {np.sum(agent2.known_walls)} cells")
    
    # Test potential composition
    from efi.core import corner_hazard
    Hc = corner_hazard(walls_mask)
    P = agent2.compose_P(walls_mask, corner_field=Hc)
    print(f"   Potential composed: shape={P.shape}, range=[{P.min():.2f}, {P.max():.2f}]")
    
    # Test valence learning
    agent2.learn_valence("B", -0.5)
    print(f"   After learning B is bad: {agent2.valence}")
    
    print("\n✓ All tests passed!")
    
    # Test 3: Validate key behaviors
    print("\n3. Testing key behaviors...")
    
    # Run multiple episodes to see valence evolution
    print("   Running 5 episodes to observe valence learning...")
    for ep in range(5):
        env.reset()
        agent2.reset()
        
        # Run episode manually to use FieldController
        obs = env.reset()
        for step in range(50):  # Short episodes
            walls_mask = agent2.step_fields(obs)
            P = agent2.compose_P(walls_mask, corner_field=corner_hazard(walls_mask))
            
            # Pick action (simplified - no temperature/momentum for test)
            from efi.core import pick_action_from_potential
            a = pick_action_from_potential(P, env.y, env.x, walls_mask)
            
            obs, r, done, info = env.step(a)
            
            # Learn from pickups
            picked = info.get("picked")
            if picked == "A":
                agent2.learn_valence("A", env_cfg.reward_A)
            elif picked == "B":
                agent2.learn_valence("B", env_cfg.reward_B)
                
            if done:
                break
                
        print(f"   Episode {ep+1}: valences = {agent2.valence}")
    
    # Verify B valence went negative
    assert agent2.valence["B"] < 0, "B valence should be negative after learning"
    print("\n✓ B avoidance learned successfully!")


if __name__ == "__main__":
    try:
        test_field_controller()
        print("\n🎉 Field controller generalization successful!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()