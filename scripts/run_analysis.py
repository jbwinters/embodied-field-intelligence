#!/usr/bin/env python3
"""Run comprehensive analysis for GitHub Pages."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

from efi.configs import EnvConfig, AgentConfig, SchemaConfig, Ablations
from efi.envs import ForageWorld
from efi.agents import ChemotaxisAgentCA, SchemaField
from efi.evaluation import run_episode, run_experiment
from efi.core import set_global_seed

# Create output directory
output_dir = Path("docs/assets/data")
output_dir.mkdir(parents=True, exist_ok=True)

print("Running EFI Analysis Suite...")
print("=" * 50)

# 1. Baseline Performance Analysis
print("\n1. Baseline Performance (10 episodes, 3 seeds)")
baseline_results = []

for seed in [42, 123, 456]:
    set_global_seed(seed)
    env_cfg = EnvConfig(H=20, W=20, win=5, p_wall=0.12, n_targets_A=3, n_targets_B=3, max_steps=200, seed=seed)
    agent_cfg = AgentConfig(seed=seed)
    ablate = Ablations()
    
    env = ForageWorld(env_cfg)
    agent = ChemotaxisAgentCA(env, agent_cfg, ablate)
    
    returns = []
    for ep in range(10):
        env.reset()
        agent.reset()
        ret, _, metrics, _ = run_episode(env, agent, None, ablate, render="none")
        returns.append(ret)
    
    baseline_results.append({
        "seed": seed,
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "returns": returns
    })
    print(f"  Seed {seed}: {np.mean(returns):.3f} ± {np.std(returns):.3f}")

# 2. Ablation Study
print("\n2. Ablation Study (5 episodes per condition)")
ablation_conditions = [
    ("Full Model", {"trail": 1, "novelty": 1, "corner": 1, "schema": 0}),
    ("No Trail", {"trail": 0, "novelty": 1, "corner": 1, "schema": 0}),
    ("No Novelty", {"trail": 1, "novelty": 0, "corner": 1, "schema": 0}),
    ("No Corner", {"trail": 1, "novelty": 1, "corner": 0, "schema": 0}),
    ("Baseline Only", {"trail": 0, "novelty": 0, "corner": 0, "schema": 0}),
]

ablation_results = []
set_global_seed(42)

for name, toggles in ablation_conditions:
    env_cfg = EnvConfig(H=20, W=20, seed=42)
    agent_cfg = AgentConfig(seed=42)
    ablate = Ablations(**toggles)
    
    env = ForageWorld(env_cfg)
    agent = ChemotaxisAgentCA(env, agent_cfg, ablate)
    
    returns = []
    for ep in range(5):
        env.reset()
        agent.reset()
        ret, _, metrics, _ = run_episode(env, agent, None, ablate, render="none")
        returns.append(ret)
    
    ablation_results.append({
        "condition": name,
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "config": toggles
    })
    print(f"  {name}: {np.mean(returns):.3f} ± {np.std(returns):.3f}")

# 3. Scaling Analysis
print("\n3. Scaling Analysis (grid size vs performance)")
grid_sizes = [10, 15, 20, 25, 30]
scaling_results = []

set_global_seed(42)
for size in grid_sizes:
    env_cfg = EnvConfig(H=size, W=size, n_targets_A=int(size*0.15), n_targets_B=int(size*0.15), seed=42)
    agent_cfg = AgentConfig(seed=42)
    ablate = Ablations()
    
    env = ForageWorld(env_cfg)
    agent = ChemotaxisAgentCA(env, agent_cfg, ablate)
    
    returns = []
    steps_list = []
    for ep in range(3):
        env.reset()
        agent.reset()
        ret, _, metrics, _ = run_episode(env, agent, None, ablate, render="none")
        returns.append(ret)
        steps_list.append(metrics.steps)
    
    scaling_results.append({
        "grid_size": size,
        "mean_return": np.mean(returns),
        "mean_steps": np.mean(steps_list),
        "targets": int(size*0.15) * 2
    })
    print(f"  {size}x{size}: Return={np.mean(returns):.3f}, Steps={np.mean(steps_list):.1f}")

# 4. Parameter Sensitivity
print("\n4. Parameter Sensitivity (diffusion rate)")
diffusion_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
sensitivity_results = []

set_global_seed(42)
for diff_rate in diffusion_rates:
    env_cfg = EnvConfig(H=20, W=20, seed=42)
    agent_cfg = AgentConfig(seed=42, scent_diff=diff_rate)
    ablate = Ablations()
    
    env = ForageWorld(env_cfg)
    agent = ChemotaxisAgentCA(env, agent_cfg, ablate)
    
    returns = []
    for ep in range(3):
        env.reset()
        agent.reset()
        ret, _, metrics, _ = run_episode(env, agent, None, ablate, render="none")
        returns.append(ret)
    
    sensitivity_results.append({
        "diffusion_rate": diff_rate,
        "mean_return": np.mean(returns)
    })
    print(f"  Diff={diff_rate:.2f}: {np.mean(returns):.3f}")

# Save all results
results = {
    "baseline": baseline_results,
    "ablation": ablation_results,
    "scaling": scaling_results,
    "sensitivity": sensitivity_results
}

with open(output_dir / "analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 50)
print("Analysis complete! Results saved to docs/assets/data/")

# Generate visualizations
print("\nGenerating visualizations...")

# 1. Ablation bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

conditions = [r["condition"] for r in ablation_results]
means = [r["mean_return"] for r in ablation_results]
stds = [r["std_return"] for r in ablation_results]

ax1.bar(range(len(conditions)), means, yerr=stds, capsize=5, color=['green', 'orange', 'orange', 'orange', 'red'])
ax1.set_xticks(range(len(conditions)))
ax1.set_xticklabels(conditions, rotation=45, ha='right')
ax1.set_ylabel('Mean Return')
ax1.set_title('Ablation Study Results')
ax1.grid(axis='y', alpha=0.3)

# 2. Scaling plot
sizes = [r["grid_size"] for r in scaling_results]
returns = [r["mean_return"] for r in scaling_results]

ax2.plot(sizes, returns, 'o-', linewidth=2, markersize=8)
ax2.set_xlabel('Grid Size')
ax2.set_ylabel('Mean Return')
ax2.set_title('Performance vs Environment Scale')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('docs/assets/images/analysis_charts.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Sensitivity plot
fig, ax = plt.subplots(figsize=(8, 5))

diff_rates = [r["diffusion_rate"] for r in sensitivity_results]
returns = [r["mean_return"] for r in sensitivity_results]

ax.plot(diff_rates, returns, 'o-', linewidth=2, markersize=8, color='purple')
ax.set_xlabel('Scent Diffusion Rate')
ax.set_ylabel('Mean Return')
ax.set_title('Parameter Sensitivity: Diffusion Rate')
ax.grid(True, alpha=0.3)
ax.axvline(x=0.25, color='red', linestyle='--', alpha=0.5, label='Default (0.25)')
ax.legend()

plt.tight_layout()
plt.savefig('docs/assets/images/sensitivity_plot.png', dpi=150, bbox_inches='tight')
plt.close()

print("Visualizations saved to docs/assets/images/")