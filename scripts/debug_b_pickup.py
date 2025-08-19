#!/usr/bin/env python3
"""Debug why agent picks up B with negative valence."""

import numpy as np
from efi.configs import EnvConfig, AgentConfig, Ablations
from efi.envs import ForageWorld
from efi.agents import ChemotaxisAgentCA
from efi.agents.field_controller import FieldController
from efi.agents.adapters import ForageAdapter
from efi.evaluation import run_episode

# Simulate environment at step 200+
env_cfg = EnvConfig(
    H=40, W=40, 
    n_targets_A=35, 
    n_targets_B=55,
    reward_A=1.0,
    reward_B=-0.5,
    max_steps=600,
    seed=6
)

agent_cfg = AgentConfig(
    valence_lr=0.25,
    valence_clip=1.5,
    valA_init=1.0,
    valB_init=0.1,
    affect_enabled=True,
    membrane_enabled=True,
    brain_membrane_enabled=True,
    seed=6
)

ablate = Ablations()

env = ForageWorld(env_cfg)

# Use FieldController as in the command
adapter = ForageAdapter(env)
agent = FieldController(env, adapter, agent_cfg, ablate)

# Simulate learned valences after 200 steps
agent.valence['A'] = 1.5  # Maxed out
agent.valence['B'] = -1.275  # Strongly negative

print("Simulated valences after 200 steps:")
print(f"  A: {agent.valence['A']:.3f}")
print(f"  B: {agent.valence['B']:.3f}")

# Reset and step to generate fields
obs = env.reset()
agent.reset()

# Step once to generate fields
agent.step_fields(obs)

# Check field strengths
GA = agent.fields['A']
GB = agent.fields['B']

print(f"\nField statistics:")
print(f"  GA max: {GA.max():.3f}, mean: {GA.mean():.6f}")
print(f"  GB max: {GB.max():.3f}, mean: {GB.mean():.6f}")

# Check weighted contributions
print(f"\nWeighted contributions:")
print(f"  A contribution (max): {agent.valence['A'] * GA.max():.3f}")
print(f"  B contribution (max): {agent.valence['B'] * GB.max():.3f}")

# Check if there are locations where B field dominates despite negative weight
if GB.max() > 0:
    # Find strongest B location
    b_max_loc = np.unravel_index(GB.argmax(), GB.shape)
    print(f"\nStrongest B location: {b_max_loc}")
    print(f"  GA at that location: {GA[b_max_loc]:.6f}")
    print(f"  GB at that location: {GB[b_max_loc]:.6f}")
    print(f"  Weighted A: {agent.valence['A'] * GA[b_max_loc]:.6f}")
    print(f"  Weighted B: {agent.valence['B'] * GB[b_max_loc]:.6f}")
    print(f"  Net attraction: {agent.valence['A'] * GA[b_max_loc] + agent.valence['B'] * GB[b_max_loc]:.6f}")
    
    # Check if agent would still approach B
    if agent.valence['B'] * GB[b_max_loc] + agent.valence['A'] * GA[b_max_loc] < 0:
        print("  -> Agent would APPROACH this B target (net negative potential)")
    else:
        print("  -> Agent would AVOID this B target (net positive potential)")

# Check exploration drive
print(f"\nNovelty weight: {agent.valence.get('Novel', agent.cfg.w_novel):.3f}")
print("Novelty can override learned aversion if exploration drive is strong.")