#!/usr/bin/env python3
"""Simple test of B field behavior."""

import numpy as np
from efi.configs import EnvConfig, AgentConfig, Ablations
from efi.envs import ForageWorld
from efi.agents import ChemotaxisAgentCA

# Create simple environment with B targets
env_cfg = EnvConfig(
    H=10, W=10, 
    n_targets_A=0,  # No A targets
    n_targets_B=2,  # Just B targets
    reward_B=-1.0,
    p_wall=0,
    seed=0
)

# Start with positive B valence
agent_cfg = AgentConfig(
    valB_init=0.5,  # Positive initial valence
    seed=0
)

ablate = Ablations()
env = ForageWorld(env_cfg)
agent = ChemotaxisAgentCA(env, agent_cfg, ablate)

print("Initial setup:")
print(f"  B targets at: {np.argwhere(env.B)}")
print(f"  Initial B valence: {agent.valence['B']:.3f}")

# Reset and step to generate fields
obs = env.reset()
action, fields = agent.step(obs)

print(f"\nWith positive B valence ({agent.valence['B']:.3f}):")
print(f"  GB field max: {fields['GB'].max():.3f}")
print(f"  GB field sum: {fields['GB'].sum():.3f}")

# Compose potential
P1 = agent.compose_P(walls_mask=env.walls)
print(f"  Potential min: {P1.min():.3f}, max: {P1.max():.3f}")

# Now make B valence negative
agent.valence['B'] = -1.0
print(f"\nChanged B valence to: {agent.valence['B']:.3f}")

# Step again to regenerate potential
action, fields = agent.step(obs)
P2 = agent.compose_P(walls_mask=env.walls)

print(f"With negative B valence ({agent.valence['B']:.3f}):")
print(f"  GB field max: {fields['GB'].max():.3f}")
print(f"  GB field sum: {fields['GB'].sum():.3f}")
print(f"  Potential min: {P2.min():.3f}, max: {P2.max():.3f}")

# Check B target locations
if fields['GB'].max() > 0:
    b_loc = np.unravel_index(fields['GB'].argmax(), fields['GB'].shape)
    print(f"\nAt B field maximum {b_loc}:")
    print(f"  With positive valence: P = {P1[b_loc]:.3f}")
    print(f"  With negative valence: P = {P2[b_loc]:.3f}")
    
    if P2[b_loc] > P1[b_loc]:
        print("  ✓ Potential increased (became repulsive)")
    else:
        print("  ✗ Potential decreased or same (still attractive)")