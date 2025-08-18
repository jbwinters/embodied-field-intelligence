#!/usr/bin/env python3
"""
Comprehensive experimental analysis for EFI research.
This script runs extensive experiments to evaluate the CA-based navigation system.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from datetime import datetime
import time
from typing import List, Dict, Tuple

from efi.configs import EnvConfig, AgentConfig, SchemaConfig, Ablations
from efi.envs import ForageWorld
from efi.agents import ChemotaxisAgentCA, SchemaField
from efi.evaluation import run_episode
from efi.core import set_global_seed

# Create output directories
output_dir = Path("docs/assets/data")
output_dir.mkdir(parents=True, exist_ok=True)
img_dir = Path("docs/assets/images")
img_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("EMBODIED FIELD INTELLIGENCE - COMPREHENSIVE ANALYSIS")
print("=" * 80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Store all results
all_results = {}

# ==============================================================================
# 1. BASELINE PERFORMANCE ACROSS MULTIPLE SEEDS
# ==============================================================================
print("1. BASELINE PERFORMANCE ANALYSIS")
print("-" * 40)

baseline_seeds = [42, 123, 456, 789, 2024]
baseline_episodes = 20
baseline_results = []

for seed in baseline_seeds:
    set_global_seed(seed)
    env_cfg = EnvConfig(H=20, W=20, win=5, p_wall=0.12, 
                       n_targets_A=3, n_targets_B=3, 
                       max_steps=200, seed=seed)
    agent_cfg = AgentConfig(seed=seed, seed_strength=1.0, 
                           scent_diff=0.25, scent_decay=0.005)
    ablate = Ablations(trail=1, novelty=1, corner=1, schema=0)
    
    env = ForageWorld(env_cfg)
    agent = ChemotaxisAgentCA(env, agent_cfg, ablate)
    
    returns = []
    steps_list = []
    targets_collected = []
    
    for ep in range(baseline_episodes):
        env.reset()
        agent.reset()
        ret, _, metrics, _ = run_episode(env, agent, None, ablate, render="none")
        returns.append(ret)
        steps_list.append(metrics.steps)
        targets_collected.append(metrics.targets_collected_A + metrics.targets_collected_B)
    
    result = {
        "seed": seed,
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "median_return": np.median(returns),
        "max_return": np.max(returns),
        "min_return": np.min(returns),
        "mean_steps": np.mean(steps_list),
        "mean_targets": np.mean(targets_collected),
        "success_rate": np.mean([r > 0 for r in returns]),
        "returns": returns
    }
    baseline_results.append(result)
    
    print(f"  Seed {seed:4d}: μ={result['mean_return']:+6.3f} σ={result['std_return']:5.3f} "
          f"Success={result['success_rate']*100:5.1f}% Targets={result['mean_targets']:.2f}")

all_results["baseline"] = baseline_results

# ==============================================================================
# 2. DETAILED ABLATION STUDY
# ==============================================================================
print("\n2. ABLATION STUDY")
print("-" * 40)

ablation_configs = [
    ("Full Model", {"trail": 1, "novelty": 1, "corner": 1, "schema": 0}),
    ("No Trail", {"trail": 0, "novelty": 1, "corner": 1, "schema": 0}),
    ("No Novelty", {"trail": 1, "novelty": 0, "corner": 1, "schema": 0}),
    ("No Corner", {"trail": 1, "novelty": 1, "corner": 0, "schema": 0}),
    ("Trail Only", {"trail": 1, "novelty": 0, "corner": 0, "schema": 0}),
    ("Novelty Only", {"trail": 0, "novelty": 1, "corner": 0, "schema": 0}),
    ("Baseline (None)", {"trail": 0, "novelty": 0, "corner": 0, "schema": 0}),
]

ablation_results = []
ablation_episodes = 15
ablation_seeds = [42, 123, 456]

for name, toggles in ablation_configs:
    all_returns = []
    all_steps = []
    all_targets = []
    
    for seed in ablation_seeds:
        set_global_seed(seed)
        env_cfg = EnvConfig(H=20, W=20, seed=seed)
        agent_cfg = AgentConfig(seed=seed)
        ablate = Ablations(**toggles)
        
        env = ForageWorld(env_cfg)
        agent = ChemotaxisAgentCA(env, agent_cfg, ablate)
        
        for ep in range(ablation_episodes // len(ablation_seeds)):
            env.reset()
            agent.reset()
            ret, _, metrics, _ = run_episode(env, agent, None, ablate, render="none")
            all_returns.append(ret)
            all_steps.append(metrics.steps)
            all_targets.append(metrics.targets_collected_A + metrics.targets_collected_B)
    
    result = {
        "condition": name,
        "mean_return": np.mean(all_returns),
        "std_return": np.std(all_returns),
        "median_return": np.median(all_returns),
        "mean_targets": np.mean(all_targets),
        "success_rate": np.mean([r > 0 for r in all_returns]),
        "config": toggles,
        "n_episodes": len(all_returns)
    }
    ablation_results.append(result)
    
    print(f"  {name:18s}: μ={result['mean_return']:+6.3f} σ={result['std_return']:5.3f} "
          f"Success={result['success_rate']*100:5.1f}% Targets={result['mean_targets']:.2f}")

all_results["ablation"] = ablation_results

# ==============================================================================
# 3. ENVIRONMENT SCALING ANALYSIS
# ==============================================================================
print("\n3. ENVIRONMENT SCALING")
print("-" * 40)

grid_configs = [
    (10, 2, 2),    # Small: 10x10, 2A, 2B
    (15, 3, 3),    # Small-Med: 15x15, 3A, 3B
    (20, 4, 4),    # Medium: 20x20, 4A, 4B
    (25, 5, 5),    # Med-Large: 25x25, 5A, 5B
    (30, 6, 6),    # Large: 30x30, 6A, 6B
    (35, 7, 7),    # X-Large: 35x35, 7A, 7B
]

scaling_results = []
scaling_episodes = 10

for size, n_a, n_b in grid_configs:
    set_global_seed(42)
    env_cfg = EnvConfig(H=size, W=size, n_targets_A=n_a, n_targets_B=n_b, 
                       max_steps=size*10, seed=42)
    agent_cfg = AgentConfig(seed=42)
    ablate = Ablations()
    
    env = ForageWorld(env_cfg)
    agent = ChemotaxisAgentCA(env, agent_cfg, ablate)
    
    returns = []
    steps_list = []
    targets_list = []
    
    for ep in range(scaling_episodes):
        env.reset()
        agent.reset()
        ret, _, metrics, _ = run_episode(env, agent, None, ablate, render="none")
        returns.append(ret)
        steps_list.append(metrics.steps)
        targets_list.append(metrics.targets_collected_A + metrics.targets_collected_B)
    
    result = {
        "grid_size": size,
        "n_targets": n_a + n_b,
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "mean_steps": np.mean(steps_list),
        "mean_targets_collected": np.mean(targets_list),
        "efficiency": np.mean(targets_list) / (n_a + n_b) if (n_a + n_b) > 0 else 0,
        "success_rate": np.mean([r > 0 for r in returns])
    }
    scaling_results.append(result)
    
    print(f"  {size:2d}x{size:2d} ({n_a+n_b:2d} targets): Return={result['mean_return']:+6.3f} "
          f"Efficiency={result['efficiency']*100:5.1f}% Success={result['success_rate']*100:5.1f}%")

all_results["scaling"] = scaling_results

# ==============================================================================
# 4. PARAMETER SENSITIVITY ANALYSIS
# ==============================================================================
print("\n4. PARAMETER SENSITIVITY")
print("-" * 40)

# Test different diffusion rates
print("  4a. Scent Diffusion Rate:")
diffusion_rates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
diff_results = []

for diff_rate in diffusion_rates:
    set_global_seed(42)
    env_cfg = EnvConfig(H=20, W=20, seed=42)
    agent_cfg = AgentConfig(seed=42, scent_diff=diff_rate)
    ablate = Ablations()
    
    env = ForageWorld(env_cfg)
    agent = ChemotaxisAgentCA(env, agent_cfg, ablate)
    
    returns = []
    for ep in range(10):
        env.reset()
        agent.reset()
        ret, _, metrics, _ = run_episode(env, agent, None, ablate, render="none")
        returns.append(ret)
    
    diff_results.append({
        "parameter": "scent_diff",
        "value": diff_rate,
        "mean_return": np.mean(returns),
        "std_return": np.std(returns)
    })
    print(f"    Diff={diff_rate:.2f}: {np.mean(returns):+6.3f} ± {np.std(returns):5.3f}")

# Test different trail decay rates
print("  4b. Trail Decay Rate:")
trail_decays = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
trail_results = []

for decay_rate in trail_decays:
    set_global_seed(42)
    env_cfg = EnvConfig(H=20, W=20, seed=42)
    agent_cfg = AgentConfig(seed=42, v_decay=decay_rate)
    ablate = Ablations()
    
    env = ForageWorld(env_cfg)
    agent = ChemotaxisAgentCA(env, agent_cfg, ablate)
    
    returns = []
    for ep in range(10):
        env.reset()
        agent.reset()
        ret, _, metrics, _ = run_episode(env, agent, None, ablate, render="none")
        returns.append(ret)
    
    trail_results.append({
        "parameter": "v_decay",
        "value": decay_rate,
        "mean_return": np.mean(returns),
        "std_return": np.std(returns)
    })
    print(f"    Decay={decay_rate:.3f}: {np.mean(returns):+6.3f} ± {np.std(returns):5.3f}")

all_results["sensitivity_diffusion"] = diff_results
all_results["sensitivity_trail"] = trail_results

# ==============================================================================
# 5. WALL DENSITY IMPACT
# ==============================================================================
print("\n5. WALL DENSITY ANALYSIS")
print("-" * 40)

wall_densities = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
wall_results = []

for p_wall in wall_densities:
    set_global_seed(42)
    env_cfg = EnvConfig(H=20, W=20, p_wall=p_wall, seed=42)
    agent_cfg = AgentConfig(seed=42)
    ablate = Ablations()
    
    env = ForageWorld(env_cfg)
    agent = ChemotaxisAgentCA(env, agent_cfg, ablate)
    
    returns = []
    for ep in range(10):
        env.reset()
        agent.reset()
        ret, _, metrics, _ = run_episode(env, agent, None, ablate, render="none")
        returns.append(ret)
    
    result = {
        "wall_density": p_wall,
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "success_rate": np.mean([r > 0 for r in returns])
    }
    wall_results.append(result)
    print(f"  p_wall={p_wall:.2f}: Return={result['mean_return']:+6.3f} "
          f"Success={result['success_rate']*100:5.1f}%")

all_results["wall_density"] = wall_results

# ==============================================================================
# SAVE ALL RESULTS
# ==============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")

# Save JSON data
with open(output_dir / "comprehensive_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Results saved to {output_dir / 'comprehensive_results.json'}")

# ==============================================================================
# GENERATE VISUALIZATIONS
# ==============================================================================
print("\nGENERATING VISUALIZATIONS")
print("-" * 40)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# 1. Baseline Performance Box Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Returns by seed
seeds = [r["seed"] for r in baseline_results]
returns_by_seed = [r["returns"] for r in baseline_results]
ax = axes[0]
bp = ax.boxplot(returns_by_seed, labels=seeds, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.set_xlabel('Random Seed')
ax.set_ylabel('Episode Return')
ax.set_title('Baseline Performance Across Seeds')
ax.grid(True, alpha=0.3)

# Success rates
success_rates = [r["success_rate"]*100 for r in baseline_results]
ax = axes[1]
ax.bar(range(len(seeds)), success_rates, color='green', alpha=0.7)
ax.set_xticks(range(len(seeds)))
ax.set_xticklabels(seeds)
ax.set_xlabel('Random Seed')
ax.set_ylabel('Success Rate (%)')
ax.set_title('Success Rate by Seed')
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)

# Targets collected
mean_targets = [r["mean_targets"] for r in baseline_results]
ax = axes[2]
ax.bar(range(len(seeds)), mean_targets, color='orange', alpha=0.7)
ax.set_xticks(range(len(seeds)))
ax.set_xticklabels(seeds)
ax.set_xlabel('Random Seed')
ax.set_ylabel('Mean Targets Collected')
ax.set_title('Average Targets per Episode')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(img_dir / 'baseline_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Ablation Study Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Mean returns with error bars
conditions = [r["condition"] for r in ablation_results]
means = [r["mean_return"] for r in ablation_results]
stds = [r["std_return"] for r in ablation_results]

ax = axes[0]
colors = ['green' if m >= -1 else 'orange' if m >= -1.5 else 'red' for m in means]
bars = ax.bar(range(len(conditions)), means, yerr=stds, capsize=5, color=colors, alpha=0.7)
ax.set_xticks(range(len(conditions)))
ax.set_xticklabels(conditions, rotation=45, ha='right')
ax.set_ylabel('Mean Return')
ax.set_title('Ablation Study: Component Contributions')
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax.grid(True, alpha=0.3)

# Success rates
success = [r["success_rate"]*100 for r in ablation_results]
ax = axes[1]
bars = ax.bar(range(len(conditions)), success, color=colors, alpha=0.7)
ax.set_xticks(range(len(conditions)))
ax.set_xticklabels(conditions, rotation=45, ha='right')
ax.set_ylabel('Success Rate (%)')
ax.set_title('Ablation Study: Success Rates')
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(img_dir / 'ablation_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Scaling Analysis
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sizes = [r["grid_size"] for r in scaling_results]
returns = [r["mean_return"] for r in scaling_results]
efficiency = [r["efficiency"]*100 for r in scaling_results]
success = [r["success_rate"]*100 for r in scaling_results]
targets = [r["n_targets"] for r in scaling_results]

# Return vs grid size
ax = axes[0, 0]
ax.plot(sizes, returns, 'o-', linewidth=2, markersize=8, color='blue')
ax.fill_between(sizes, 
                [r["mean_return"] - r["std_return"] for r in scaling_results],
                [r["mean_return"] + r["std_return"] for r in scaling_results],
                alpha=0.3)
ax.set_xlabel('Grid Size')
ax.set_ylabel('Mean Return')
ax.set_title('Performance vs Environment Scale')
ax.grid(True, alpha=0.3)

# Efficiency
ax = axes[0, 1]
ax.plot(sizes, efficiency, 'o-', linewidth=2, markersize=8, color='green')
ax.set_xlabel('Grid Size')
ax.set_ylabel('Target Collection Efficiency (%)')
ax.set_title('Collection Efficiency vs Scale')
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)

# Success rate
ax = axes[1, 0]
ax.plot(sizes, success, 'o-', linewidth=2, markersize=8, color='orange')
ax.set_xlabel('Grid Size')
ax.set_ylabel('Success Rate (%)')
ax.set_title('Success Rate vs Scale')
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)

# Targets vs grid size
ax = axes[1, 1]
ax.bar(range(len(sizes)), targets, color='purple', alpha=0.7)
ax.set_xticks(range(len(sizes)))
ax.set_xticklabels([f"{s}×{s}" for s in sizes])
ax.set_xlabel('Grid Size')
ax.set_ylabel('Total Targets')
ax.set_title('Target Count by Grid Size')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(img_dir / 'scaling_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. Parameter Sensitivity
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Diffusion rate sensitivity
ax = axes[0]
diff_vals = [r["value"] for r in diff_results]
diff_means = [r["mean_return"] for r in diff_results]
diff_stds = [r["std_return"] for r in diff_results]
ax.errorbar(diff_vals, diff_means, yerr=diff_stds, marker='o', linewidth=2, 
           markersize=8, capsize=5, color='purple')
ax.axvline(x=0.25, color='red', linestyle='--', alpha=0.5, label='Default')
ax.set_xlabel('Scent Diffusion Rate')
ax.set_ylabel('Mean Return')
ax.set_title('Sensitivity: Diffusion Rate')
ax.legend()
ax.grid(True, alpha=0.3)

# Trail decay sensitivity
ax = axes[1]
trail_vals = [r["value"] for r in trail_results]
trail_means = [r["mean_return"] for r in trail_results]
trail_stds = [r["std_return"] for r in trail_results]
ax.errorbar(trail_vals, trail_means, yerr=trail_stds, marker='o', linewidth=2,
           markersize=8, capsize=5, color='brown')
ax.axvline(x=0.02, color='red', linestyle='--', alpha=0.5, label='Default')
ax.set_xlabel('Trail Decay Rate')
ax.set_ylabel('Mean Return')
ax.set_title('Sensitivity: Trail Decay')
ax.legend()
ax.grid(True, alpha=0.3)

# Wall density impact
ax = axes[2]
wall_vals = [r["wall_density"] for r in wall_results]
wall_means = [r["mean_return"] for r in wall_results]
wall_success = [r["success_rate"]*100 for r in wall_results]
ax2 = ax.twinx()
l1 = ax.plot(wall_vals, wall_means, 'o-', linewidth=2, markersize=8, 
            color='blue', label='Return')
l2 = ax2.plot(wall_vals, wall_success, 's-', linewidth=2, markersize=8, 
             color='red', label='Success %')
ax.set_xlabel('Wall Density')
ax.set_ylabel('Mean Return', color='blue')
ax2.set_ylabel('Success Rate (%)', color='red')
ax.set_title('Impact of Wall Density')
ax.tick_params(axis='y', labelcolor='blue')
ax2.tick_params(axis='y', labelcolor='red')
lines = l1 + l2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(img_dir / 'parameter_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()

print("Visualizations saved to docs/assets/images/")

# ==============================================================================
# STATISTICAL SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("STATISTICAL SUMMARY")
print("-" * 40)

# Overall baseline statistics
all_baseline_returns = []
for r in baseline_results:
    all_baseline_returns.extend(r["returns"])

print(f"Baseline Performance (n={len(all_baseline_returns)} episodes):")
print(f"  Mean Return: {np.mean(all_baseline_returns):+.3f}")
print(f"  Std Dev:     {np.std(all_baseline_returns):.3f}")
print(f"  Median:      {np.median(all_baseline_returns):+.3f}")
print(f"  Success Rate: {np.mean([r > 0 for r in all_baseline_returns])*100:.1f}%")

# Best configuration from ablation
best_ablation = max(ablation_results, key=lambda x: x["mean_return"])
print(f"\nBest Ablation Configuration: {best_ablation['condition']}")
print(f"  Mean Return: {best_ablation['mean_return']:+.3f}")

# Optimal parameters
best_diff = max(diff_results, key=lambda x: x["mean_return"])
print(f"\nOptimal Diffusion Rate: {best_diff['value']:.2f}")
print(f"  Mean Return: {best_diff['mean_return']:+.3f}")

best_trail = max(trail_results, key=lambda x: x["mean_return"])
print(f"\nOptimal Trail Decay: {best_trail['value']:.3f}")
print(f"  Mean Return: {best_trail['mean_return']:+.3f}")

print("\n" + "=" * 80)
print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)