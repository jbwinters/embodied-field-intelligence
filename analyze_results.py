#!/usr/bin/env python3
"""
Analyze experimental results and generate visualizations for GitHub Pages.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# Directories
data_dir = Path("docs/assets/data")
img_dir = Path("docs/assets/images")
img_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("ANALYZING EXPERIMENTAL RESULTS")
print("=" * 80)

# ==============================================================================
# 1. LOAD AND PROCESS ABLATION DATA
# ==============================================================================
print("\n1. ABLATION STUDY ANALYSIS")
print("-" * 40)

with open(data_dir / "ablation_suite/suite_summary.json") as f:
    ablation_data = json.load(f)

for result in ablation_data:
    print(f"  {result['condition']:10s}: μ={result['mean']:+6.3f} σ={result['std']:5.3f} (n={result['n']})")

# Create ablation visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

conditions = [r['condition'] for r in ablation_data]
means = [r['mean'] for r in ablation_data]
stds = [r['std'] for r in ablation_data]

# Bar chart with error bars
colors = ['#2ecc71' if c == 'full' else '#e74c3c' if c == '-trail' else '#f39c12' for c in conditions]
bars = ax1.bar(range(len(conditions)), means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
ax1.set_xticks(range(len(conditions)))
ax1.set_xticklabels(conditions, fontsize=12)
ax1.set_ylabel('Mean Episode Return', fontsize=12)
ax1.set_title('Ablation Study: Component Contributions', fontsize=14, fontweight='bold')
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar, mean, std in zip(bars, means, stds):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + (std if height > 0 else -std-0.1),
             f'{mean:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)

# Component impact (difference from full model)
full_mean = next(r['mean'] for r in ablation_data if r['condition'] == 'full')
impacts = [full_mean - r['mean'] if r['condition'] != 'full' else 0 for r in ablation_data]
colors2 = ['gray' if i == 0 else '#e74c3c' if i > 1 else '#f39c12' for i in impacts]
bars2 = ax2.bar(range(len(conditions)), impacts, color=colors2, alpha=0.8, edgecolor='black')
ax2.set_xticks(range(len(conditions)))
ax2.set_xticklabels(conditions, fontsize=12)
ax2.set_ylabel('Performance Loss (vs Full Model)', fontsize=12)
ax2.set_title('Component Impact Analysis', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Add value labels
for bar, impact in zip(bars2, impacts):
    if impact != 0:
        ax2.text(bar.get_x() + bar.get_width()/2., impact + 0.05,
                f'{impact:.2f}', ha='center', va='bottom', fontsize=10)

plt.suptitle('EFI Ablation Study Results', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(img_dir / 'ablation_results.png', dpi=150, bbox_inches='tight')
plt.close()

# ==============================================================================
# 2. SCALING ANALYSIS
# ==============================================================================
print("\n2. SCALING ANALYSIS")
print("-" * 40)

scale_results = []
for size in [10, 15, 20, 25, 30]:
    csv_path = data_dir / f"scale_{size}/eval_returns.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        returns = df['return'].values
        scale_results.append({
            'size': size,
            'mean': np.mean(returns),
            'std': np.std(returns),
            'median': np.median(returns),
            'success_rate': np.mean(returns > 0) * 100,
            'n': len(returns)
        })
        print(f"  {size:2d}x{size:2d}: μ={np.mean(returns):+6.3f} σ={np.std(returns):5.3f} "
              f"Success={np.mean(returns > 0)*100:5.1f}% (n={len(returns)})")

# Create scaling visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

sizes = [r['size'] for r in scale_results]
means = [r['mean'] for r in scale_results]
stds = [r['std'] for r in scale_results]
success_rates = [r['success_rate'] for r in scale_results]

# Performance vs size
ax = axes[0, 0]
ax.errorbar(sizes, means, yerr=stds, marker='o', linewidth=2, markersize=10, 
           capsize=5, color='#3498db', label='Mean ± Std')
ax.fill_between(sizes, [m-s for m,s in zip(means, stds)], 
                [m+s for m,s in zip(means, stds)], alpha=0.2, color='#3498db')
ax.set_xlabel('Grid Size', fontsize=12)
ax.set_ylabel('Mean Episode Return', fontsize=12)
ax.set_title('Performance vs Environment Scale', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

# Success rate vs size
ax = axes[0, 1]
ax.plot(sizes, success_rates, 'o-', linewidth=2, markersize=10, color='#27ae60')
ax.set_xlabel('Grid Size', fontsize=12)
ax.set_ylabel('Success Rate (%)', fontsize=12)
ax.set_title('Success Rate vs Environment Scale', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)

# Return distribution by size
ax = axes[1, 0]
all_returns = []
labels = []
for size in [10, 15, 20, 25, 30]:
    csv_path = data_dir / f"scale_{size}/eval_returns.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        all_returns.append(df['return'].values)
        labels.append(f"{size}×{size}")

bp = ax.boxplot(all_returns, labels=labels, patch_artist=True)
for patch, color in zip(bp['boxes'], plt.cm.viridis(np.linspace(0.3, 0.9, len(all_returns)))):
    patch.set_facecolor(color)
ax.set_xlabel('Grid Size', fontsize=12)
ax.set_ylabel('Episode Return', fontsize=12)
ax.set_title('Return Distribution by Grid Size', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)

# Efficiency (normalized performance)
ax = axes[1, 1]
# Normalize by number of targets
normalized_means = [m / (s/5 * 2) for m, s in zip(means, sizes)]  # Each size has size/5 A and B targets
ax.bar(range(len(sizes)), normalized_means, color=plt.cm.plasma(np.linspace(0.3, 0.9, len(sizes))))
ax.set_xticks(range(len(sizes)))
ax.set_xticklabels([f"{s}×{s}" for s in sizes])
ax.set_xlabel('Grid Size', fontsize=12)
ax.set_ylabel('Return per Target', fontsize=12)
ax.set_title('Normalized Efficiency', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.suptitle('Environment Scaling Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(img_dir / 'scaling_results.png', dpi=150, bbox_inches='tight')
plt.close()

# ==============================================================================
# 3. PARAMETER SENSITIVITY
# ==============================================================================
print("\n3. PARAMETER SENSITIVITY (Diffusion Rate)")
print("-" * 40)

diff_results = []
for diff in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
    csv_path = data_dir / f"diff_{diff}/eval_returns.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        returns = df['return'].values
        diff_results.append({
            'diff': diff,
            'mean': np.mean(returns),
            'std': np.std(returns),
            'n': len(returns)
        })
        print(f"  Diff={diff:4.2f}: μ={np.mean(returns):+6.3f} σ={np.std(returns):5.3f}")

# Create sensitivity visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

diffs = [r['diff'] for r in diff_results]
means = [r['mean'] for r in diff_results]
stds = [r['std'] for r in diff_results]

# Performance vs diffusion rate
ax = axes[0]
ax.errorbar(diffs, means, yerr=stds, marker='o', linewidth=2, markersize=10,
           capsize=5, color='#9b59b6', label='Mean ± Std')
ax.axvline(x=0.25, color='red', linestyle='--', alpha=0.5, label='Default (0.25)')
ax.set_xlabel('Scent Diffusion Rate', fontsize=12)
ax.set_ylabel('Mean Episode Return', fontsize=12)
ax.set_title('Sensitivity to Diffusion Rate', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

# Variance analysis
ax = axes[1]
ax.plot(diffs, stds, 'o-', linewidth=2, markersize=10, color='#e67e22')
ax.set_xlabel('Scent Diffusion Rate', fontsize=12)
ax.set_ylabel('Standard Deviation', fontsize=12)
ax.set_title('Performance Variance vs Diffusion Rate', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.suptitle('Parameter Sensitivity Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(img_dir / 'sensitivity_results.png', dpi=150, bbox_inches='tight')
plt.close()

# ==============================================================================
# 4. BASELINE PERFORMANCE
# ==============================================================================
print("\n4. BASELINE PERFORMANCE")
print("-" * 40)

baseline_path = data_dir / "baseline/eval_returns.csv"
if baseline_path.exists():
    df = pd.read_csv(baseline_path)
    returns = df['return'].values
    seeds = df['seed'].values
    episodes = df['episode'].values
    
    print(f"  Total episodes: {len(returns)}")
    print(f"  Mean return: {np.mean(returns):+.3f} ± {np.std(returns):.3f}")
    print(f"  Median return: {np.median(returns):+.3f}")
    print(f"  Success rate: {np.mean(returns > 0)*100:.1f}%")
    print(f"  Max return: {np.max(returns):+.3f}")
    print(f"  Min return: {np.min(returns):+.3f}")
    
    # Create baseline visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Return distribution
    ax = axes[0, 0]
    ax.hist(returns, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    ax.axvline(x=np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.2f}')
    ax.axvline(x=np.median(returns), color='green', linestyle='--', label=f'Median: {np.median(returns):.2f}')
    ax.set_xlabel('Episode Return', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Return Distribution (Baseline)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Returns by seed
    ax = axes[0, 1]
    unique_seeds = np.unique(seeds)
    seed_returns = [returns[seeds == s] for s in unique_seeds]
    bp = ax.boxplot(seed_returns, labels=unique_seeds, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#27ae60')
    ax.set_xlabel('Random Seed', fontsize=12)
    ax.set_ylabel('Episode Return', fontsize=12)
    ax.set_title('Performance by Seed', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Learning curve (moving average)
    ax = axes[1, 0]
    window = 10
    moving_avg = pd.Series(returns).rolling(window=window).mean()
    ax.plot(moving_avg, linewidth=2, color='#e74c3c', label=f'{window}-episode moving average')
    ax.fill_between(range(len(returns)), 
                    pd.Series(returns).rolling(window=window).mean() - pd.Series(returns).rolling(window=window).std(),
                    pd.Series(returns).rolling(window=window).mean() + pd.Series(returns).rolling(window=window).std(),
                    alpha=0.2, color='#e74c3c')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Return', fontsize=12)
    ax.set_title('Performance Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Success rate over time
    ax = axes[1, 1]
    success = (returns > 0).astype(int)
    success_rate = pd.Series(success).rolling(window=window).mean() * 100
    ax.plot(success_rate, linewidth=2, color='#f39c12')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title(f'Success Rate ({window}-episode window)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Baseline Performance Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(img_dir / 'baseline_results.png', dpi=150, bbox_inches='tight')
    plt.close()

# ==============================================================================
# 5. SUMMARY STATISTICS
# ==============================================================================
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("-" * 40)

summary = {
    "ablation": {
        "best_config": min(ablation_data, key=lambda x: abs(x['mean']))['condition'],
        "worst_config": min(ablation_data, key=lambda x: x['mean'])['condition'],
        "trail_impact": next(r['mean'] for r in ablation_data if r['condition'] == 'full') - 
                       next(r['mean'] for r in ablation_data if r['condition'] == '-trail')
    },
    "scaling": {
        "best_size": max(scale_results, key=lambda x: x['mean'])['size'] if scale_results else None,
        "best_performance": max(scale_results, key=lambda x: x['mean'])['mean'] if scale_results else None
    },
    "sensitivity": {
        "optimal_diffusion": max(diff_results, key=lambda x: x['mean'])['diff'] if diff_results else None,
        "optimal_performance": max(diff_results, key=lambda x: x['mean'])['mean'] if diff_results else None
    }
}

print(f"Best Ablation Config: {summary['ablation']['best_config']}")
print(f"Trail Field Impact: {summary['ablation']['trail_impact']:+.3f} return")
print(f"Best Grid Size: {summary['scaling']['best_size']}x{summary['scaling']['best_size']}")
print(f"Optimal Diffusion Rate: {summary['sensitivity']['optimal_diffusion']}")

# Save summary
with open(data_dir / "experiment_summary.json", "w") as f:
    json.dump({
        "ablation_results": ablation_data,
        "scale_results": scale_results,
        "sensitivity_results": diff_results,
        "summary": summary
    }, f, indent=2)

print(f"\nResults saved to {data_dir / 'experiment_summary.json'}")
print("Visualizations saved to docs/assets/images/")
print("=" * 80)