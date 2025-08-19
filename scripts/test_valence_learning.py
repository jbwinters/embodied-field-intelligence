#!/usr/bin/env python3
"""Test valence learning behavior."""

from efi.configs import EnvConfig, AgentConfig, Ablations
from efi.envs import ForageWorld
from efi.agents import ChemotaxisAgentCA
from efi.evaluation import run_episode

# Simulate similar environment
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
agent = ChemotaxisAgentCA(env, agent_cfg, ablate)

print("Initial valences:")
print(f"  A: {agent.valence['A']:.3f}")
print(f"  B: {agent.valence['B']:.3f}")

# Track valence evolution
A_picks = 0
B_picks = 0
steps = 0

obs = env.reset()
for t in range(200):  # First 200 steps
    action, _ = agent.step(obs)
    obs, reward, done, info = env.step(action)
    
    if info.get('picked') == 'A':
        A_picks += 1
        prev_val = agent.valence['B']
        agent.learn_valence("A", env.cfg.reward_A)
        print(f"Step {t}: Picked A, valA: {agent.valence['A']:.3f}")
    elif info.get('picked') == 'B':
        B_picks += 1
        prev_val = agent.valence['B']
        agent.learn_valence("B", env.cfg.reward_B)
        print(f"Step {t}: Picked B, valB: {agent.valence['B']:.3f} (was {prev_val:.3f})")
    
    steps += 1
    if done:
        break

print(f"\nAfter {steps} steps:")
print(f"  A picks: {A_picks}")
print(f"  B picks: {B_picks}")
print(f"  Final valA: {agent.valence['A']:.3f}")
print(f"  Final valB: {agent.valence['B']:.3f}")

# Check if B is still attractive
if agent.valence['B'] > -0.5:
    print(f"\nWARNING: B valence ({agent.valence['B']:.3f}) is still positive or weakly negative!")
    print("Agent may still be attracted to B targets.")