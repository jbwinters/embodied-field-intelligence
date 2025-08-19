#!/usr/bin/env python3
"""Focused test of wall proximity effect."""

import numpy as np
from efi.configs import EnvConfig, AgentConfig, Ablations
from efi.envs import ForageWorld
from efi.agents import ChemotaxisAgentCA
from efi.core import wall_proximity_field

# Create a simple box environment
cfg = EnvConfig(H=10, W=10, p_wall=0.0, n_targets_A=1, n_targets_B=0)
env = ForageWorld(cfg)
env.reset()

# Create walls around the edges
env.walls[:] = False
env.walls[0, :] = True  # Top wall
env.walls[-1, :] = True  # Bottom wall
env.walls[:, 0] = True  # Left wall
env.walls[:, -1] = True  # Right wall

# Put agent near a wall
env.y, env.x = 1, 1  # Near top-left corner

# Create agent with wall proximity enabled
agent_cfg = AgentConfig()
agent_cfg.w_wall_prox = 0.5
agent = ChemotaxisAgentCA(env, agent_cfg, Ablations(wall_proximity=True))
agent.reset()

# Take a step to discover walls
obs_vec = env._obs()
_, fields = agent.step(obs_vec)

print("Environment with walls around edges:")
print("Agent at position (1, 1) near corner")
print()

# Show ASCII map
for y in range(env.H):
    row = []
    for x in range(env.W):
        if env.walls[y, x]:
            row.append('#')
        elif env.y == y and env.x == x:
            row.append('@')
        else:
            row.append('.')
    print(''.join(row))

print("\nWall Proximity Field values:")
print("(Should be high near walls, decreasing with distance)")

# Compute and display wall proximity
W_prox = wall_proximity_field(agent.known_walls, radius=1.5)

# Sample some positions
positions = [
    (1, 1, "Near corner"),
    (1, 5, "Near top wall center"),
    (5, 5, "Center of room"),
    (8, 1, "Near bottom-left"),
]

for y, x, desc in positions:
    print(f"  ({y},{x}) {desc}: {W_prox[y, x]:.3f}")

print("\nField values at agent position (1,1):")
print(f"  WallProx from fields: {fields['WallProx'][1,1]:.3f}")
print(f"  Trail: {fields['Vtrail'][1,1]:.3f}")
print(f"  Known walls discovered: {np.sum(agent.known_walls)}")

# Now test with agent that hasn't discovered walls yet
print("\n" + "="*50)
print("Testing with fresh agent (no walls discovered):")
agent2 = ChemotaxisAgentCA(env, agent_cfg, Ablations(wall_proximity=True))
agent2.reset()
# Don't take a step - walls remain undiscovered
print(f"  Known walls: {np.sum(agent2.known_walls)}")
W_prox2 = wall_proximity_field(agent2.known_walls, radius=1.5)
print(f"  WallProx at (1,1): {W_prox2[1,1]:.3f}")
print("  (Should be 0 since no walls discovered yet)")