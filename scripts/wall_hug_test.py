#!/usr/bin/env python3
"""Test wall-hugging behavior with and without wall proximity field."""

import numpy as np
from efi.configs import EnvConfig, AgentConfig, SchemaConfig, Ablations
from efi.envs import ForageWorld
from efi.agents import ChemotaxisAgentCA
from efi.core import corner_hazard, pick_action_from_potential


def run_corridor_test(use_wall_prox=False, show_steps=False):
    """Run agent through a corridor and count wall bumps."""
    
    # Create corridor environment
    cfg = EnvConfig(H=12, W=25, p_wall=0.0, n_targets_A=1, n_targets_B=0, max_steps=40)
    env = ForageWorld(cfg)
    env.reset()
    
    # Create L-shaped corridor
    env.walls[:] = True
    # Horizontal section
    env.walls[5:7, 1:15] = False
    # Vertical section turning upward
    env.walls[2:6, 14:16] = False
    # Continue horizontal at top
    env.walls[2:4, 15:23] = False
    
    # Place target at end
    env.TA[:] = False
    env.TA[2, 22] = True
    
    # Start at beginning
    env.y, env.x = 5, 1
    
    # Configure agent
    agent_cfg = AgentConfig(seed=42)
    if use_wall_prox:
        agent_cfg.w_wall_prox = 0.4  # Moderate wall repulsion
    
    agent = ChemotaxisAgentCA(env, agent_cfg, Ablations(wall_proximity=use_wall_prox))
    agent.reset()
    
    bumps = 0
    wall_adjacent_steps = 0
    path = [(env.y, env.x)]
    
    for step in range(40):
        obs_vec = env._obs()
        _, fields = agent.step(obs_vec)
        
        # Check if adjacent to wall
        y, x = env.y, env.x
        adjacent_to_wall = False
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < env.H and 0 <= nx < env.W and env.walls[ny, nx]:
                adjacent_to_wall = True
                break
        if adjacent_to_wall:
            wall_adjacent_steps += 1
        
        # Compose potential
        P_eff = agent.compose_P(
            walls_mask=env.walls,
            corner_field=corner_hazard(env.walls),
            wall_prox_field=fields.get('WallProx') if use_wall_prox else None,
            schema_bias=None,
            frontier_weight=0.15
        )
        
        # Choose action
        a = pick_action_from_potential(P_eff, env.y, env.x, env.walls, temperature=0.1)
        
        # Show state periodically
        if show_steps and step % 8 == 0:
            print(f"\nStep {step}:")
            for row_y in range(env.H):
                row = []
                for col_x in range(env.W):
                    if env.walls[row_y, col_x]:
                        row.append('#')
                    elif env.y == row_y and env.x == col_x:
                        row.append('@')
                    elif env.TA[row_y, col_x]:
                        row.append('A')
                    elif (row_y, col_x) in path:
                        row.append('.')
                    else:
                        row.append(' ')
                print(''.join(row))
            
            if use_wall_prox:
                print(f"  WallProx at position: {fields['WallProx'][env.y, env.x]:.3f}")
        
        obs, r, done, info = env.step(a)
        path.append((env.y, env.x))
        
        if not info.get('moved', True):
            bumps += 1
        
        if done:
            break
    
    return bumps, wall_adjacent_steps, len(path), env.TA[env.y, env.x] == False


print("="*60)
print("WALL-HUGGING TEST: L-Shaped Corridor Navigation")
print("="*60)

# Run without wall proximity
print("\nWITHOUT Wall Proximity Field:")
bumps1, adjacent1, steps1, reached1 = run_corridor_test(use_wall_prox=False, show_steps=True)
print(f"\nResults:")
print(f"  Wall bumps: {bumps1}")
print(f"  Steps adjacent to wall: {adjacent1}/{steps1} ({100*adjacent1/steps1:.1f}%)")
print(f"  Target reached: {'Yes' if reached1 else 'No'}")

print("\n" + "="*60)

# Run with wall proximity
print("\nWITH Wall Proximity Field:")
bumps2, adjacent2, steps2, reached2 = run_corridor_test(use_wall_prox=True, show_steps=True)
print(f"\nResults:")
print(f"  Wall bumps: {bumps2}")
print(f"  Steps adjacent to wall: {adjacent2}/{steps2} ({100*adjacent2/steps2:.1f}%)")
print(f"  Target reached: {'Yes' if reached2 else 'No'}")

print("\n" + "="*60)
print("COMPARISON:")
print(f"  Bump reduction: {bumps1} -> {bumps2} ({bumps1-bumps2} fewer)")
print(f"  Wall adjacency: {100*adjacent1/steps1:.1f}% -> {100*adjacent2/steps2:.1f}%")
print("="*60)