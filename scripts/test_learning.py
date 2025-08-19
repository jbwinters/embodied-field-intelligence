#!/usr/bin/env python3
"""Test learning behaviors - valence and schema."""

import numpy as np
from efi.configs import EnvConfig, AgentConfig, SchemaConfig, Ablations
from efi.envs import ForageWorld
from efi.agents import ChemotaxisAgentCA, SchemaField
from efi.evaluation import run_episode

print("="*60)
print("TEST: Aversive Learning (B targets become repulsive)")
print("="*60)

# Create environment with clear good/bad targets
cfg = EnvConfig(
    H=12, W=12,
    n_targets_A=3,  # Good targets
    n_targets_B=6,  # Bad targets (more of them)
    reward_A=1.0,   # Positive reward
    reward_B=-1.0,  # Negative reward
    step_cost=-0.01,
    max_steps=150,
    p_wall=0.05
)

# Run multiple episodes to see learning
for episode in range(3):
    print(f"\n--- Episode {episode+1} ---")
    
    env = ForageWorld(cfg)
    
    if episode == 0:
        # Fresh agent
        agent_cfg = AgentConfig(
            valA_init=0.2,  # Slightly positive
            valB_init=0.2,  # Start thinking B is also good
            valence_lr=0.3  # Learn from pickups
        )
        agent_cfg.valence_lr_step = 0.002  # Counterfactual learning
        agent = ChemotaxisAgentCA(env, agent_cfg, Ablations())
        schema_cfg = SchemaConfig(K=4, enabled=True)
        schema_cfg.rho_valence = 0.05  # Set as attribute
        schema = SchemaField(env.H, env.W, feature_dim=6, cfg=schema_cfg)
    
    print(f"Starting valences: A={agent.valence['A']:.3f}, B={agent.valence['B']:.3f}")
    
    # Run episode
    total_return, _, metrics, _ = run_episode(env, agent, schema, Ablations())
    
    print(f"Episode results:")
    print(f"  Return: {total_return:.2f}")
    print(f"  A collected: {metrics.targets_collected['A']}")
    print(f"  B collected: {metrics.targets_collected['B']}")
    print(f"Final valences: A={agent.valence['A']:.3f}, B={agent.valence['B']:.3f}")
    
    # Check schema learning
    if episode > 0:
        neg_count = np.sum(schema.q < -0.01)
        pos_count = np.sum(schema.q > 0.01)
        print(f"Schema prototypes: {pos_count} positive, {neg_count} negative")

print("\n" + "="*60)
print("TEST: Field Aggregation Modes")
print("="*60)

# Create environment with multiple signals
cfg = EnvConfig(H=10, W=10, n_targets_A=2, n_targets_B=2, p_wall=0.0)
env = ForageWorld(cfg)
env.reset()

# Place targets in specific positions
env.TA[:] = False
env.TB[:] = False
env.TA[2, 2] = True  # A in top-left
env.TA[7, 7] = True  # A in bottom-right
env.TB[2, 7] = True  # B in top-right
env.TB[7, 2] = True  # B in bottom-left

agent = ChemotaxisAgentCA(env, AgentConfig(), Ablations())
agent.reset()

# Take a step to seed scents
obs = env._obs()
_, fields = agent.step(obs)

# Test different aggregation modes
from efi.core import compose_potential

print("\nField values at center (5,5):")
print(f"  GA: {fields['GA'][5,5]:.3f}")
print(f"  GB: {fields['GB'][5,5]:.3f}")

attractors = {"A": fields['GA'], "B": fields['GB']}
repulsors = {"Trail": fields['Vtrail']}
w_attr = {"A": 1.0, "B": 0.5}
w_rep = {"Trail": 0.3}

modes = [
    ("linear", "Linear (sum)"),
    ("lse", "Log-sum-exp (soft max)"),
    ("maxplus", "Max-plus (winner takes all)")
]

print("\nPotential at center with different modes:")
for mode, desc in modes:
    P = compose_potential(attractors, repulsors, w_attr, w_rep, 
                         mode=mode, beta_attr=2.0)
    print(f"  {desc}: {P[5,5]:.3f}")

print("\n" + "="*60)