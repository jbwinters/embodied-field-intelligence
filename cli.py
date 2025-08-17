#!/usr/bin/env python3
"""Command-line interface for EFI experiments."""

import argparse
import json
import csv
from pathlib import Path
from dataclasses import asdict

import numpy as np

from efi.configs import EnvConfig, AgentConfig, SchemaConfig, Ablations
from efi.core import set_global_seed, ts, ensure_dir
from efi.envs import ForageWorld, register_gym_env
from efi.agents import ChemotaxisAgentCA, SchemaField
from efi.evaluation import run_episode, run_experiment
from efi.visualization import save_video_mp4, plot_experiment_results
from efi.visualization.interactive import create_interactive_viewer
from efi.visualization.html_viewer import create_html_viewer


def add_common_args(parser):
    """Add common arguments to parser."""
    # Environment
    parser.add_argument("--H", type=int, default=17, help="Grid height")
    parser.add_argument("--W", type=int, default=17, help="Grid width")
    parser.add_argument("--win", type=int, default=5, help="Observation window size")
    parser.add_argument("--p-wall", type=float, default=0.12, dest="p_wall", help="Wall probability")
    parser.add_argument("--nA", type=int, default=3, help="Number of A targets")
    parser.add_argument("--nB", type=int, default=3, help="Number of B targets")
    parser.add_argument("--max-steps", type=int, default=200, dest="max_steps", help="Max episode steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    # Agent
    parser.add_argument("--seed-strength", type=float, default=0.6, dest="seed_strength")
    parser.add_argument("--scent-diff", type=float, default=0.14, dest="scent_diff")
    parser.add_argument("--scent-decay", type=float, default=0.01, dest="scent_decay")
    parser.add_argument("--scent-steps", type=int, default=2, dest="scent_steps")
    parser.add_argument("--v-inj", type=float, default=1.0, dest="v_inj")
    parser.add_argument("--v-decay", type=float, default=0.03, dest="v_decay")
    parser.add_argument("--v-diff", type=float, default=0.10, dest="v_diff")
    parser.add_argument("--k-repulse", type=float, default=0.30, dest="k_repulse")
    parser.add_argument("--wander", type=float, default=0.08)
    parser.add_argument("--stay-thresh", type=float, default=0.02, dest="stay_thresh")
    parser.add_argument("--anti-stuck-after", type=int, default=3, dest="anti_stuck_after")
    parser.add_argument("--anti-stuck-temp", type=float, default=0.6, dest="anti_stuck_temp")
    parser.add_argument("--internal-think", type=int, default=0, dest="internal_think")
    
    # Ablations
    parser.add_argument("--novelty", type=int, default=1, choices=[0,1])
    parser.add_argument("--trail", type=int, default=1, choices=[0,1])
    parser.add_argument("--corner", type=int, default=1, choices=[0,1])
    parser.add_argument("--schema", type=int, default=1, choices=[0,1])
    
    # Schema
    parser.add_argument("--schema-tile", type=int, default=5, dest="schema_tile")
    parser.add_argument("--schema-K", type=int, default=4, dest="schema_K")
    parser.add_argument("--schema-eta", type=float, default=0.03, dest="schema_eta")
    parser.add_argument("--schema-slowness", type=float, default=0.02, dest="schema_slowness")
    parser.add_argument("--schema-bcm-tau", type=float, default=0.01, dest="schema_bcm_tau")
    parser.add_argument("--schema-comp-k", type=int, default=1, dest="schema_comp_k")
    parser.add_argument("--schema-diff", type=float, default=0.08, dest="schema_diff")
    parser.add_argument("--schema-decay", type=float, default=0.005, dest="schema_decay")
    parser.add_argument("--schema-steps", type=int, default=1, dest="schema_steps")
    parser.add_argument("--schema-alpha", type=float, default=0.35, dest="schema_alpha")
    
    # Output
    parser.add_argument("--out", type=str, default=None, help="Output directory")


def run_demo(args):
    """Run demo mode."""
    set_global_seed(args.seed)
    out_dir = ensure_dir(args.out or f"runs/demo-{ts()}")
    
    # Create configs
    env_cfg = EnvConfig(
        H=args.H, W=args.W, win=args.win, p_wall=args.p_wall,
        n_targets_A=args.nA, n_targets_B=args.nB,
        max_steps=args.max_steps, seed=args.seed
    )
    
    agent_cfg = AgentConfig(
        seed_strength=args.seed_strength, scent_diff=args.scent_diff,
        scent_decay=args.scent_decay, scent_steps=args.scent_steps,
        v_inj=args.v_inj, v_decay=args.v_decay, v_diff=args.v_diff,
        k_repulse=args.k_repulse, wander=args.wander, stay_thresh=args.stay_thresh,
        anti_stuck_after=args.anti_stuck_after, anti_stuck_temp=args.anti_stuck_temp,
        internal_think=args.internal_think, seed=args.seed
    )
    
    ablate = Ablations(
        trail=args.trail, novelty=args.novelty, 
        corner=args.corner, schema=args.schema
    )
    
    # Create environment and agent
    env = ForageWorld(env_cfg)
    agent = ChemotaxisAgentCA(env, agent_cfg, ablate)
    
    # Create schema if enabled
    schema = None
    if args.schema:
        schema_cfg = SchemaConfig(
            enabled=1, tile=args.schema_tile, K=args.schema_K, eta=args.schema_eta,
            slowness=args.schema_slowness, bcm_tau=args.schema_bcm_tau,
            comp_k=args.schema_comp_k, diff=args.schema_diff, decay=args.schema_decay,
            steps=args.schema_steps, alpha_schema=args.schema_alpha, seed=args.seed
        )
        schema = SchemaField(env.H, env.W, 6, schema_cfg)
    
    # Run episodes
    returns = []
    last_frames = []
    
    for ep in range(args.episodes):
        ret, frames, metrics, _ = run_episode(
            env, agent, schema, ablate, 
            render=args.render, 
            record=bool(args.save_video),
            record_fields=False
        )
        returns.append(ret)
        last_frames = frames
        print(f"[demo] episode {ep+1}/{args.episodes} | return={ret:+.3f}")
    
    # Save video if requested
    if args.save_video and len(last_frames) > 0:
        save_video_mp4(last_frames, Path(args.save_video), fps=8)
    
    # Save metrics
    meta = dict(
        env=asdict(env_cfg),
        agent=asdict(agent_cfg),
        schema=(asdict(schema.cfg) if schema else None),
        ablate=asdict(ablate),
        returns=returns
    )
    
    with open(out_dir / "demo_metrics.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"[demo] metrics saved at {out_dir/'demo_metrics.json'}")


def run_eval(args):
    """Run evaluation mode."""
    set_global_seed(args.seed)
    out_dir = ensure_dir(args.out or f"runs/eval-{ts()}")
    
    # Create configs
    env_cfg = EnvConfig(
        H=args.H, W=args.W, win=args.win, p_wall=args.p_wall,
        n_targets_A=args.nA, n_targets_B=args.nB,
        max_steps=args.max_steps, seed=args.seed
    )
    
    agent_cfg = AgentConfig(
        seed_strength=args.seed_strength, scent_diff=args.scent_diff,
        scent_decay=args.scent_decay, scent_steps=args.scent_steps,
        v_inj=args.v_inj, v_decay=args.v_decay, v_diff=args.v_diff,
        k_repulse=args.k_repulse, wander=args.wander, stay_thresh=args.stay_thresh,
        anti_stuck_after=args.anti_stuck_after, anti_stuck_temp=args.anti_stuck_temp,
        internal_think=args.internal_think, seed=args.seed
    )
    
    schema_cfg = None
    if args.schema:
        schema_cfg = SchemaConfig(
            enabled=1, tile=args.schema_tile, K=args.schema_K, eta=args.schema_eta,
            slowness=args.schema_slowness, bcm_tau=args.schema_bcm_tau,
            comp_k=args.schema_comp_k, diff=args.schema_diff, decay=args.schema_decay,
            steps=args.schema_steps, alpha_schema=args.schema_alpha, seed=args.seed
        )
    
    ablate = Ablations(
        trail=args.trail, novelty=args.novelty,
        corner=args.corner, schema=args.schema
    )
    
    # Run experiment
    results = run_experiment(
        env_cfg, agent_cfg, schema_cfg, ablate,
        episodes=args.episodes, seeds=args.seeds, base_seed=args.seed
    )
    
    # Save results
    rows = []
    for m in results.metrics:
        rows.append({
            "seed": m.seed,
            "episode": m.episode,
            "return": m.total_return,
            "steps": m.steps,
            "efficiency": m.efficiency
        })
    
    # Save CSV
    csv_path = out_dir / "eval_returns.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "episode", "return", "steps", "efficiency"])
        w.writeheader()
        w.writerows(rows)
    
    # Save JSON
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump({
            "config": results.config,
            "metrics": rows,
            "summary": {
                "mean_return": results.mean_return,
                "std_return": results.std_return,
                "mean_steps": results.mean_steps,
                "std_steps": results.std_steps
            }
        }, f, indent=2)
    
    # Plot results
    fig = plot_experiment_results(results, save_path=out_dir / "results_plot.png")
    
    print(f"[eval] saved {len(rows)} results to {csv_path}")
    print(f"[eval] mean return: {results.mean_return:.3f} ± {results.std_return:.3f}")


def run_suite(args):
    """Run ablation suite."""
    set_global_seed(args.seed)
    out_dir = ensure_dir(args.out or f"runs/suite-{ts()}")
    
    # Define ablation conditions
    suites = [
        ("full",    dict(trail=1, novelty=1, corner=1, schema=1)),
        ("-trail",  dict(trail=0, novelty=1, corner=1, schema=1)),
        ("-novel",  dict(trail=1, novelty=0, corner=1, schema=1)),
        ("-corner", dict(trail=1, novelty=1, corner=0, schema=1)),
        ("-schema", dict(trail=1, novelty=1, corner=1, schema=0)),
    ]
    
    # Base configs
    env_cfg = EnvConfig(
        H=args.H, W=args.W, win=args.win, p_wall=args.p_wall,
        n_targets_A=args.nA, n_targets_B=args.nB,
        max_steps=args.max_steps, seed=args.seed
    )
    
    agent_cfg = AgentConfig(
        seed_strength=args.seed_strength, scent_diff=args.scent_diff,
        scent_decay=args.scent_decay, scent_steps=args.scent_steps,
        v_inj=args.v_inj, v_decay=args.v_decay, v_diff=args.v_diff,
        k_repulse=args.k_repulse, wander=args.wander, stay_thresh=args.stay_thresh,
        anti_stuck_after=args.anti_stuck_after, anti_stuck_temp=args.anti_stuck_temp,
        internal_think=args.internal_think, seed=args.seed
    )
    
    schema_cfg = SchemaConfig(
        enabled=1, tile=args.schema_tile, K=args.schema_K, eta=args.schema_eta,
        slowness=args.schema_slowness, bcm_tau=args.schema_bcm_tau,
        comp_k=args.schema_comp_k, diff=args.schema_diff, decay=args.schema_decay,
        steps=args.schema_steps, alpha_schema=args.schema_alpha, seed=args.seed
    )
    
    # Run each condition
    suite_results = []
    
    for label, toggles in suites:
        print(f"[suite] running {label}...")
        ablate = Ablations(**toggles)
        
        results = run_experiment(
            env_cfg, agent_cfg, 
            schema_cfg if toggles.get("schema", 1) else None,
            ablate,
            episodes=args.episodes, 
            seeds=args.seeds, 
            base_seed=args.seed
        )
        
        suite_results.append({
            "condition": label,
            "mean": results.mean_return,
            "std": results.std_return,
            "n": len(results.metrics)
        })
        
        print(f"[suite] {label}: mean={results.mean_return:+.3f} ± {results.std_return:.3f}")
    
    # Save summary
    with open(out_dir / "suite_summary.json", "w") as f:
        json.dump(suite_results, f, indent=2)
    
    print(f"[suite] summary saved at {out_dir/'suite_summary.json'}")


def run_gym_register(args):
    """Register Gymnasium environment."""
    env_cfg = EnvConfig(
        H=args.H, W=args.W, win=args.win, p_wall=args.p_wall,
        n_targets_A=args.nA, n_targets_B=args.nB,
        max_steps=args.max_steps, seed=args.seed
    )
    register_gym_env(env_cfg)


def build_parser():
    """Build argument parser."""
    parser = argparse.ArgumentParser(description="EFI - Embodied Field Intelligence")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    
    # Demo mode
    demo_parser = subparsers.add_parser("demo", help="Run demo episodes")
    add_common_args(demo_parser)
    demo_parser.add_argument("--episodes", type=int, default=5)
    demo_parser.add_argument("--render", type=str, default="none", choices=["none", "live"])
    demo_parser.add_argument("--save-video", type=str, default=None, dest="save_video")
    
    # Eval mode
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    add_common_args(eval_parser)
    eval_parser.add_argument("--episodes", type=int, default=50)
    eval_parser.add_argument("--seeds", type=int, default=3)
    
    # Suite mode
    suite_parser = subparsers.add_parser("suite", help="Run ablation suite")
    add_common_args(suite_parser)
    suite_parser.add_argument("--episodes", type=int, default=20)
    suite_parser.add_argument("--seeds", type=int, default=5)
    
    # Gym register
    gym_parser = subparsers.add_parser("gym-register", help="Register Gymnasium environment")
    add_common_args(gym_parser)
    
    # Interactive viewer mode
    interactive_parser = subparsers.add_parser("interactive", help="Run episode with interactive viewer")
    add_common_args(interactive_parser)
    interactive_parser.add_argument("--auto-play", action="store_true", help="Auto-play on start")
    
    return parser


def run_interactive(args):
    """Run interactive viewer mode."""
    set_global_seed(args.seed)
    
    # Create configs
    env_cfg = EnvConfig(
        H=args.H, W=args.W, win=args.win, p_wall=args.p_wall,
        n_targets_A=args.nA, n_targets_B=args.nB,
        max_steps=args.max_steps, seed=args.seed
    )
    
    agent_cfg = AgentConfig(
        seed_strength=args.seed_strength, scent_diff=args.scent_diff,
        scent_decay=args.scent_decay, scent_steps=args.scent_steps,
        v_inj=args.v_inj, v_decay=args.v_decay, v_diff=args.v_diff,
        k_repulse=args.k_repulse, wander=args.wander, stay_thresh=args.stay_thresh,
        anti_stuck_after=args.anti_stuck_after, anti_stuck_temp=args.anti_stuck_temp,
        internal_think=args.internal_think, seed=args.seed
    )
    
    ablate = Ablations(
        trail=args.trail, novelty=args.novelty, 
        corner=args.corner, schema=args.schema
    )
    
    # Create environment and agent
    env = ForageWorld(env_cfg)
    agent = ChemotaxisAgentCA(env, agent_cfg, ablate)
    
    # Create schema if enabled
    schema = None
    if args.schema:
        schema_cfg = SchemaConfig(
            enabled=1, tile=args.schema_tile, K=args.schema_K, eta=args.schema_eta,
            slowness=args.schema_slowness, bcm_tau=args.schema_bcm_tau,
            comp_k=args.schema_comp_k, diff=args.schema_diff, decay=args.schema_decay,
            steps=args.schema_steps, alpha_schema=args.schema_alpha, seed=args.seed
        )
        schema = SchemaField(env.H, env.W, 6, schema_cfg)
    
    # Run episode with field recording
    print("[interactive] Running episode and collecting data...")
    ret, _, metrics, episode_data = run_episode(
        env, agent, schema, ablate, 
        render="none", 
        record=False,
        record_fields=True
    )
    
    print(f"[interactive] Episode complete. Return: {ret:+.3f}, Steps: {metrics.steps}")
    
    # Create and show interactive viewer
    if episode_data:
        # Try matplotlib viewer first, fall back to HTML if no display
        try:
            import matplotlib
            backend = matplotlib.get_backend()
            if 'Agg' in backend or 'pdf' in backend or 'svg' in backend:
                # Non-interactive backend, use HTML viewer
                raise RuntimeError("No display backend")
                
            print("[interactive] Launching interactive viewer...")
            viewer = create_interactive_viewer(episode_data)
            
            if args.auto_play:
                viewer.playing = True
                viewer.btn_play.label.set_text('Pause')
                
            viewer.show()
            
        except Exception as e:
            # Fall back to HTML viewer
            print("[interactive] No display detected, creating HTML viewer...")
            out_dir = ensure_dir(args.out or "runs")
            html_path = create_html_viewer(episode_data, out_dir / f"interactive_{ts()}.html")
            print(f"[interactive] HTML viewer saved to: {html_path}")
            print(f"[interactive] Open this file in a web browser to interact with the episode")
            
            # Try to open in browser if possible
            try:
                import webbrowser
                webbrowser.open(f"file://{html_path}")
                print("[interactive] Attempting to open in default browser...")
            except:
                pass
    else:
        print("[interactive] No episode data recorded.")


def main():
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()
    
    if args.mode == "demo":
        run_demo(args)
    elif args.mode == "eval":
        run_eval(args)
    elif args.mode == "suite":
        run_suite(args)
    elif args.mode == "gym-register":
        run_gym_register(args)
    elif args.mode == "interactive":
        run_interactive(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()