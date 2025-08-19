#!/usr/bin/env python3
"""Test that B becomes repulsor with negative valence."""

import numpy as np
from efi.configs import EnvConfig, AgentConfig, Ablations
from efi.envs import ForageWorld
from efi.agents.field_controller import FieldController
from efi.agents.adapters import ForageAdapter

# Create environment
env_cfg = EnvConfig(
    H=20, W=20, 
    n_targets_A=5, 
    n_targets_B=10,
    reward_A=1.0,
    reward_B=-0.5,
    seed=42
)

agent_cfg = AgentConfig(
    valence_lr=0.25,
    valA_init=1.0,
    valB_init=0.1,
    affect_enabled=True,
    seed=42
)

ablate = Ablations()

env = ForageWorld(env_cfg)
adapter = ForageAdapter(env)
agent = FieldController(env, adapter, agent_cfg, ablate)

# Test with positive B valence
print("=== Test 1: B with positive valence ===")
agent.valence['B'] = 0.5
obs = env.reset()
agent.reset()
# Step multiple times to let fields diffuse
for _ in range(5):
    agent.step_fields(obs)

# Compose potential
P = agent.compose_P(walls_mask=env.walls)

print(f"B valence: {agent.valence['B']:.3f}")
print(f"B field max: {agent.fields['B'].max():.3f}")
print(f"Potential at B max: {P[agent.fields['B'] == agent.fields['B'].max()][0] if agent.fields['B'].max() > 0 else 0:.3f}")
print(f"B treated as: {'attractor' if agent.valence['B'] >= 0 else 'repulsor'}")

# Test with negative B valence
print("\n=== Test 2: B with negative valence ===")
agent.valence['B'] = -1.0
agent.reset()
for _ in range(5):
    agent.step_fields(obs)

# Compose potential again
P2 = agent.compose_P(walls_mask=env.walls)

print(f"B valence: {agent.valence['B']:.3f}")
print(f"B field max: {agent.fields['B'].max():.3f}")
if agent.fields['B'].max() > 0:
    b_max_loc = np.unravel_index(agent.fields['B'].argmax(), agent.fields['B'].shape)
    print(f"Potential at B max location: {P2[b_max_loc]:.3f}")
    
    # Check surrounding potentials
    y, x = b_max_loc
    surrounding = []
    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
        if 0 <= y+dy < env.H and 0 <= x+dx < env.W and not env.walls[y+dy, x+dx]:
            surrounding.append(P2[y+dy, x+dx])
    
    if surrounding:
        avg_surrounding = np.mean(surrounding)
        print(f"Average potential around B: {avg_surrounding:.3f}")
        if P2[b_max_loc] > avg_surrounding:
            print("✓ B location has HIGHER potential (repulsive)")
        else:
            print("✗ B location has LOWER potential (still attractive!)")

print(f"B treated as: {'attractor' if agent.valence['B'] >= 0 else 'repulsor'}")