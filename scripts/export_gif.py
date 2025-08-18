#!/usr/bin/env python3
"""
Standalone GIF exporter for EFI episodes.

This script allows you to export GIFs from episodes even in headless environments.
Run an episode first with cli.py, then use this script to export GIFs.
"""

import argparse
import pickle
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from efi.configs import EnvConfig, AgentConfig, Ablations
from efi.envs import ForageWorld
from efi.agents import ChemotaxisAgentCA
from efi.evaluation import run_episode
from efi.core import set_global_seed


def export_full_gif(frames, world_frames, output_path, fps=8):
    """Export full multi-panel GIF."""
    n_frames = len(frames)
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f"EFI Episode - {n_frames} frames", fontsize=12)
    
    # Create grid layout
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1])
    
    # Create axes
    axes = {}
    axes['world'] = fig.add_subplot(gs[0, 0])
    axes['GA'] = fig.add_subplot(gs[0, 1])
    axes['GB'] = fig.add_subplot(gs[0, 2])
    axes['P_eff'] = fig.add_subplot(gs[1, 0])
    axes['Vtrail'] = fig.add_subplot(gs[1, 1])
    axes['info'] = fig.add_subplot(gs[1, 2])
    
    # Initialize images
    ims = {}
    
    # World view
    ax = axes['world']
    world = world_frames[0] if world_frames else np.zeros((10,10,3), dtype=np.uint8)
    ims['world'] = ax.imshow(world)
    ax.set_title("World", fontsize=10)
    ax.axis('off')
    
    if frames:
        frame = frames[0]
        
        # GA field
        ax = axes['GA']
        ims['GA'] = ax.imshow(frame.get('GA', np.zeros((10,10))), cmap='Greens', vmin=0, vmax=1)
        ax.set_title("GA (A scent)", fontsize=10)
        ax.axis('off')
        
        # GB field  
        ax = axes['GB']
        ims['GB'] = ax.imshow(frame.get('GB', np.zeros((10,10))), cmap='Purples', vmin=0, vmax=1)
        ax.set_title("GB (B scent)", fontsize=10)
        ax.axis('off')
        
        # P_eff field
        ax = axes['P_eff']
        ims['P_eff'] = ax.imshow(frame.get('P_eff', np.zeros((10,10))), cmap='plasma')
        ax.set_title("Effective Potential", fontsize=10)
        ax.axis('off')
        
        # Visit trail
        ax = axes['Vtrail']
        ims['Vtrail'] = ax.imshow(frame.get('Vtrail', np.zeros((10,10))), cmap='Oranges', vmin=0, vmax=1)
        ax.set_title("Visit Trail", fontsize=10)
        ax.axis('off')
    
    # Info panel
    ax = axes['info']
    ax.axis('off')
    info_text = ax.text(0.05, 0.5, "", fontsize=9, transform=ax.transAxes,
                       verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    
    # Animation update function
    def update(frame_idx):
        # Update world
        if frame_idx < len(world_frames):
            ims['world'].set_data(world_frames[frame_idx])
        
        # Update fields
        if frame_idx < len(frames):
            frame = frames[frame_idx]
            
            for field_name in ['GA', 'GB', 'P_eff', 'Vtrail']:
                if field_name in frame and field_name in ims:
                    ims[field_name].set_data(frame[field_name])
            
            # Update info text
            info = frame.get('info', {})
            info_lines = [
                f"Frame: {frame_idx + 1}/{n_frames}",
                f"Step: {info.get('step', frame_idx)}",
                f"Return: {info.get('return', 0.0):+.3f}",
                f"Action: {info.get('action', 'N/A')}",
            ]
            info_text.set_text('\n'.join(info_lines))
        
        return list(ims.values()) + [info_text]
    
    # Create animation
    anim = FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=int(1000.0 / fps),
        blit=True
    )
    
    # Save as GIF
    print(f"Exporting full GIF to {output_path}...")
    writer = PillowWriter(fps=fps)
    anim.save(str(output_path), writer=writer, dpi=80)
    plt.close(fig)
    
    file_size = output_path.stat().st_size / 1024
    print(f"✓ Full GIF exported ({file_size:.1f} KB)")


def export_simple_gif(frames, world_frames, output_path, fps=8):
    """Export simple world-only GIF."""
    n_frames = len(frames)
    
    # Create figure
    fig = plt.figure(figsize=(6, 6))
    
    # Single axis for world view
    ax = fig.add_subplot(111)
    world = world_frames[0] if world_frames else np.zeros((10,10,3), dtype=np.uint8)
    im = ax.imshow(world)
    ax.set_title("EFI Agent Navigation", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add frame counter
    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Animation update function
    def update(frame_idx):
        # Update world
        if frame_idx < len(world_frames):
            im.set_data(world_frames[frame_idx])
        
        # Update frame counter
        if frame_idx < len(frames):
            info = frames[frame_idx].get('info', {})
            frame_text.set_text(f"Step {frame_idx+1}/{n_frames} | Score: {info.get('return', 0.0):+.2f}")
        
        return [im, frame_text]
    
    # Create animation
    anim = FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=int(1000.0 / fps),
        blit=True
    )
    
    # Save as GIF
    print(f"Exporting simple GIF to {output_path}...")
    writer = PillowWriter(fps=fps)
    anim.save(str(output_path), writer=writer, dpi=60)
    plt.close(fig)
    
    file_size = output_path.stat().st_size / 1024
    print(f"✓ Simple GIF exported ({file_size:.1f} KB)")


def run_and_export(args):
    """Run an episode and export GIFs."""
    set_global_seed(args.seed)
    
    # Create configs
    env_cfg = EnvConfig(
        H=args.H, W=args.W, win=args.win, p_wall=args.p_wall,
        n_targets_A=args.nA, n_targets_B=args.nB,
        max_steps=args.max_steps, seed=args.seed
    )
    
    agent_cfg = AgentConfig(seed=args.seed)
    ablate = Ablations()
    
    # Create environment and agent
    env = ForageWorld(env_cfg)
    agent = ChemotaxisAgentCA(env, agent_cfg, ablate)
    
    # Run episode
    print(f"Running episode (seed={args.seed}, H={args.H}, W={args.W}, steps={args.max_steps})...")
    print(f"Targets: A={args.nA}, B={args.nB}")
    
    import time
    start_time = time.time()
    
    ret, _, metrics, episode_data = run_episode(
        env, agent, None, ablate,
        render="none", record=True, record_fields=True
    )
    
    elapsed = time.time() - start_time
    print(f"Episode complete in {elapsed:.1f}s. Return: {ret:+.3f}, Steps: {metrics.steps}")
    
    # Export GIFs
    if episode_data and episode_data.get('frames') and episode_data.get('world_frames'):
        frames = episode_data['frames']
        world_frames = episode_data['world_frames']
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if args.mode in ['full', 'both']:
            output_path = output_dir / f"efi_full_{timestamp}.gif"
            export_full_gif(frames, world_frames, output_path, args.fps)
        
        if args.mode in ['simple', 'both']:
            output_path = output_dir / f"efi_simple_{timestamp}.gif"
            export_simple_gif(frames, world_frames, output_path, args.fps)
    else:
        print("No episode data to export!")


def main():
    parser = argparse.ArgumentParser(description="Export EFI episodes as GIFs")
    
    # Episode parameters
    parser.add_argument("--seed", type=int, default=6, help="Random seed")
    parser.add_argument("--H", type=int, default=20, help="Grid height")
    parser.add_argument("--W", type=int, default=20, help="Grid width")
    parser.add_argument("--win", type=int, default=5, help="Observation window")
    parser.add_argument("--p-wall", type=float, default=0.12, help="Wall probability")
    parser.add_argument("--nA", type=int, default=2, help="Number of A targets")
    parser.add_argument("--nB", type=int, default=2, help="Number of B targets")
    parser.add_argument("--max-steps", type=int, default=200, help="Max episode steps")
    
    # Export parameters
    parser.add_argument("--mode", choices=['full', 'simple', 'both'], default='both',
                       help="Export mode: full (all fields), simple (world only), or both")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second")
    parser.add_argument("--output-dir", default="exports", help="Output directory")
    
    args = parser.parse_args()
    run_and_export(args)


if __name__ == "__main__":
    main()