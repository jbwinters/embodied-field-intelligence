"""Plotting utilities for visualization."""

from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt

from ..evaluation.metrics import ExperimentResults


def plot_frame_panels(
    world_rgb: np.ndarray, 
    GA: np.ndarray, 
    GB: np.ndarray, 
    P: np.ndarray, 
    Vtrail: np.ndarray, 
    Novel: np.ndarray, 
    Ssum: np.ndarray, 
    title: Optional[str] = None
):
    """
    Plot multi-panel visualization of agent state.
    
    Args:
        world_rgb: RGB image of world
        GA: Target A scent field
        GB: Target B scent field
        P: Effective potential field
        Vtrail: Visit trail field
        Novel: Novelty field
        Ssum: Sum of schema activations
        title: Optional figure title
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(14, 8))
    
    if title:
        fig.suptitle(title, y=0.98, fontsize=14)
    
    # World view
    ax = plt.subplot(2, 4, 1)
    ax.imshow(world_rgb)
    ax.set_title("World")
    ax.axis('off')
    
    # Scent fields
    ax = plt.subplot(2, 4, 2)
    ax.imshow(GA, cmap='Greens')
    ax.set_title("GA (A scent)")
    ax.axis('off')
    
    ax = plt.subplot(2, 4, 3)
    ax.imshow(GB, cmap='Purples')
    ax.set_title("GB (B scent)")
    ax.axis('off')
    
    # Potential field
    ax = plt.subplot(2, 4, 4)
    ax.imshow(P, cmap='plasma')
    ax.set_title("P_eff (drive)")
    ax.axis('off')
    
    # Trail and novelty
    ax = plt.subplot(2, 4, 5)
    ax.imshow(Vtrail, cmap='Oranges')
    ax.set_title("Visit-trail (repel)")
    ax.axis('off')
    
    ax = plt.subplot(2, 4, 6)
    ax.imshow(Novel, cmap='Blues')
    ax.set_title("Novelty (attract)")
    ax.axis('off')
    
    # Schema field
    ax = plt.subplot(2, 4, 7)
    ax.imshow(Ssum, cmap='cividis')
    ax.set_title("Schema sum (slow)")
    ax.axis('off')
    
    # Legend
    ax = plt.subplot(2, 4, 8)
    ax.axis('off')
    ax.text(0.0, 0.90, "Legend", fontsize=12, fontweight='bold')
    ax.text(0.0, 0.70, "Walls: dark gray", fontsize=11)
    ax.text(0.0, 0.55, "A: green, B: magenta", fontsize=11)
    ax.text(0.0, 0.40, "Trail: orange", fontsize=11)
    ax.text(0.0, 0.25, "Novelty: blue", fontsize=11)
    ax.text(0.0, 0.10, "Schema sum: cividis", fontsize=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_experiment_results(results: ExperimentResults, save_path: Optional[str] = None):
    """
    Plot experiment results summary.
    
    Args:
        results: Experiment results
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Extract data
    returns = [m.total_return for m in results.metrics]
    steps = [m.steps for m in results.metrics]
    efficiencies = [m.efficiency for m in results.metrics]
    
    # Plot returns
    ax = axes[0, 0]
    ax.hist(returns, bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(results.mean_return, color='red', linestyle='--', label=f'Mean: {results.mean_return:.2f}')
    ax.set_xlabel('Episode Return')
    ax.set_ylabel('Count')
    ax.set_title('Return Distribution')
    ax.legend()
    
    # Plot steps
    ax = axes[0, 1]
    ax.hist(steps, bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(results.mean_steps, color='red', linestyle='--', label=f'Mean: {results.mean_steps:.1f}')
    ax.set_xlabel('Episode Steps')
    ax.set_ylabel('Count')
    ax.set_title('Steps Distribution')
    ax.legend()
    
    # Plot efficiency
    ax = axes[1, 0]
    ax.hist(efficiencies, bins=20, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Efficiency (Return/Steps)')
    ax.set_ylabel('Count')
    ax.set_title('Efficiency Distribution')
    
    # Plot learning curve
    ax = axes[1, 1]
    episodes_per_seed = len(returns) // max(1, len(set(m.seed for m in results.metrics)))
    for seed in set(m.seed for m in results.metrics):
        seed_returns = [m.total_return for m in results.metrics if m.seed == seed]
        ax.plot(seed_returns, alpha=0.5, label=f'Seed {seed}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.set_title('Learning Curves')
    ax.legend()
    
    plt.suptitle('Experiment Results Summary', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig