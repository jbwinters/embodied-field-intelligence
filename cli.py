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
from efi.agents import ChemotaxisAgentCA, SchemaField, FieldController, ForageAdapter
from efi.evaluation import run_episode, run_experiment
from efi.visualization import save_video_mp4, plot_experiment_results
from efi.visualization.interactive import create_interactive_viewer
from efi.visualization.html_viewer import create_html_viewer


def add_common_args(parser):
    """Add common arguments to parser."""
    # Controller selection
    parser.add_argument("--controller", type=str, default="chemotaxis",
                       choices=["chemotaxis", "field"],
                       help="Controller type: chemotaxis (original) or field (generalized)")
    
    # Environment
    parser.add_argument("--H", type=int, default=17, help="Grid height")
    parser.add_argument("--W", type=int, default=17, help="Grid width")
    parser.add_argument("--win", type=int, default=5, help="Observation window size")
    parser.add_argument("--p-wall", type=float, default=0.12, dest="p_wall", help="Wall probability")
    parser.add_argument("--nA", type=int, default=2, help="Number of A targets")
    parser.add_argument("--nB", type=int, default=4, help="Number of B targets")
    parser.add_argument("--max-steps", type=int, default=200, dest="max_steps", help="Max episode steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    # Rewards (for learning B avoidance)
    parser.add_argument("--reward-A", type=float, default=1.0, dest="reward_A", help="Reward for collecting A")
    parser.add_argument("--reward-B", type=float, default=-0.5, dest="reward_B", help="Reward for collecting B (negative = undesirable)")
    
    # Agent (updated defaults to match improved config)
    parser.add_argument("--valA-init", type=float, default=1.0, dest="valA_init", help="Initial valence for A")
    parser.add_argument("--valB-init", type=float, default=0.1, dest="valB_init", help="Initial valence for B")
    parser.add_argument("--valence-lr", type=float, default=0.25, dest="valence_lr", help="Valence learning rate")
    parser.add_argument("--valence-clip", type=float, default=1.5, dest="valence_clip", help="Max absolute valence")
    parser.add_argument("--seed-strength", type=float, default=1.0, dest="seed_strength")
    parser.add_argument("--scent-diff", type=float, default=0.25, dest="scent_diff")
    parser.add_argument("--scent-decay", type=float, default=0.005, dest="scent_decay")
    parser.add_argument("--scent-steps", type=int, default=4, dest="scent_steps")
    parser.add_argument("--v-inj", type=float, default=1.0, dest="v_inj")
    parser.add_argument("--v-decay", type=float, default=0.02, dest="v_decay")
    parser.add_argument("--v-diff", type=float, default=0.08, dest="v_diff")
    parser.add_argument("--k-repulse", type=float, default=0.30, dest="k_repulse")
    parser.add_argument("--wander", type=float, default=0.0)
    parser.add_argument("--stay-thresh", type=float, default=0.02, dest="stay_thresh")
    parser.add_argument("--anti-stuck-after", type=int, default=2, dest="anti_stuck_after")
    parser.add_argument("--anti-stuck-temp", type=float, default=0.8, dest="anti_stuck_temp")
    parser.add_argument("--internal-think", type=int, default=0, dest="internal_think")
    
    # Ablations
    parser.add_argument("--novelty", type=int, default=1, choices=[0,1])
    parser.add_argument("--trail", type=int, default=1, choices=[0,1])
    parser.add_argument("--corner", type=int, default=1, choices=[0,1])
    parser.add_argument("--schema", type=int, default=1, choices=[0,1])
    
    # Affect system parameters (Phase 1)
    parser.add_argument("--affect-enabled", type=lambda x: x.lower() == 'true', default=False,
                       dest="affect_enabled", help="Enable affect system (pain, arousal, valence)")
    parser.add_argument("--membrane-enabled", type=lambda x: x.lower() == 'true', default=False,
                       dest="membrane_enabled", help="Enable protective membrane fields")
    parser.add_argument("--brain-membrane-enabled", type=lambda x: x.lower() == 'true', default=False,
                       dest="brain_membrane_enabled", help="Enable learning protection under stress")
    
    # Pain parameters
    parser.add_argument("--w-pain", type=float, default=0.7, dest="w_pain",
                       help="Pain field weight as repulsor")
    parser.add_argument("--pain-to-temp-gain", type=float, default=0.6, dest="pain_to_temp_gain",
                       help="Pain to temperature conversion gain")
    parser.add_argument("--pain-semiring-threshold", type=float, default=0.6, dest="pain_semiring_threshold",
                       help="Pain threshold to switch to max-plus mode")
    
    # Membrane parameters
    parser.add_argument("--w-membrane", type=float, default=0.6, dest="w_membrane",
                       help="Membrane field weight")
    parser.add_argument("--membrane-r-min", type=float, default=1.0, dest="membrane_r_min",
                       help="Minimum membrane radius")
    parser.add_argument("--membrane-r-gain-arousal", type=float, default=1.0, dest="membrane_r_gain_arousal",
                       help="Arousal contribution to membrane radius")
    parser.add_argument("--membrane-r-gain-pain", type=float, default=1.5, dest="membrane_r_gain_pain",
                       help="Pain contribution to membrane radius")
    
    # Affect EWMA rates
    parser.add_argument("--affect-rho-v", type=float, default=0.02, dest="affect_rho_v",
                       help="Valence EWMA decay rate")
    parser.add_argument("--affect-rho-a", type=float, default=0.05, dest="affect_rho_a",
                       help="Arousal EWMA decay rate")
    parser.add_argument("--affect-rho-c", type=float, default=0.05, dest="affect_rho_c",
                       help="Control EWMA decay rate")
    parser.add_argument("--affect-rho-p", type=float, default=0.1, dest="affect_rho_p",
                       help="Pain EWMA decay rate")
    
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
        max_steps=args.max_steps, seed=args.seed,
        reward_A=args.reward_A, reward_B=args.reward_B
    )
    
    agent_cfg = AgentConfig(
        seed_strength=args.seed_strength, scent_diff=args.scent_diff,
        scent_decay=args.scent_decay, scent_steps=args.scent_steps,
        v_inj=args.v_inj, v_decay=args.v_decay, v_diff=args.v_diff,
        k_repulse=args.k_repulse, wander=args.wander, stay_thresh=args.stay_thresh,
        anti_stuck_after=args.anti_stuck_after, anti_stuck_temp=args.anti_stuck_temp,
        internal_think=args.internal_think, seed=args.seed,
        # Valence learning
        valA_init=args.valA_init, valB_init=args.valB_init,
        valence_lr=args.valence_lr, valence_clip=args.valence_clip,
        # Affect system parameters
        affect_enabled=args.affect_enabled,
        membrane_enabled=args.membrane_enabled,
        brain_membrane_enabled=args.brain_membrane_enabled,
        w_pain=args.w_pain,
        pain_to_temp_gain=args.pain_to_temp_gain,
        pain_semiring_threshold=args.pain_semiring_threshold,
        w_membrane=args.w_membrane,
        membrane_r_min=args.membrane_r_min,
        membrane_r_gain_arousal=args.membrane_r_gain_arousal,
        membrane_r_gain_pain=args.membrane_r_gain_pain,
        affect_rho_v=args.affect_rho_v,
        affect_rho_a=args.affect_rho_a,
        affect_rho_c=args.affect_rho_c,
        affect_rho_p=args.affect_rho_p
    )
    
    ablate = Ablations(
        trail=args.trail, novelty=args.novelty, 
        corner=args.corner, schema=args.schema
    )
    
    # Create environment and agent
    env = ForageWorld(env_cfg)
    if hasattr(args, 'controller') and args.controller == "field":
        print("[interactive] Using FieldController")
        adapter = ForageAdapter(env)
        agent = FieldController(env, adapter, agent_cfg, ablate)
    else:
        print("[interactive] Using ChemotaxisAgentCA")
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
        max_steps=args.max_steps, seed=args.seed,
        reward_A=args.reward_A, reward_B=args.reward_B
    )
    
    agent_cfg = AgentConfig(
        seed_strength=args.seed_strength, scent_diff=args.scent_diff,
        scent_decay=args.scent_decay, scent_steps=args.scent_steps,
        v_inj=args.v_inj, v_decay=args.v_decay, v_diff=args.v_diff,
        k_repulse=args.k_repulse, wander=args.wander, stay_thresh=args.stay_thresh,
        anti_stuck_after=args.anti_stuck_after, anti_stuck_temp=args.anti_stuck_temp,
        internal_think=args.internal_think, seed=args.seed,
        # Valence learning
        valA_init=args.valA_init, valB_init=args.valB_init,
        valence_lr=args.valence_lr, valence_clip=args.valence_clip,
        # Affect system parameters
        affect_enabled=args.affect_enabled,
        membrane_enabled=args.membrane_enabled,
        brain_membrane_enabled=args.brain_membrane_enabled,
        w_pain=args.w_pain,
        pain_to_temp_gain=args.pain_to_temp_gain,
        pain_semiring_threshold=args.pain_semiring_threshold,
        w_membrane=args.w_membrane,
        membrane_r_min=args.membrane_r_min,
        membrane_r_gain_arousal=args.membrane_r_gain_arousal,
        membrane_r_gain_pain=args.membrane_r_gain_pain,
        affect_rho_v=args.affect_rho_v,
        affect_rho_a=args.affect_rho_a,
        affect_rho_c=args.affect_rho_c,
        affect_rho_p=args.affect_rho_p
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
        max_steps=args.max_steps, seed=args.seed,
        reward_A=args.reward_A, reward_B=args.reward_B
    )
    
    agent_cfg = AgentConfig(
        seed_strength=args.seed_strength, scent_diff=args.scent_diff,
        scent_decay=args.scent_decay, scent_steps=args.scent_steps,
        v_inj=args.v_inj, v_decay=args.v_decay, v_diff=args.v_diff,
        k_repulse=args.k_repulse, wander=args.wander, stay_thresh=args.stay_thresh,
        anti_stuck_after=args.anti_stuck_after, anti_stuck_temp=args.anti_stuck_temp,
        internal_think=args.internal_think, seed=args.seed,
        # Valence learning
        valA_init=args.valA_init, valB_init=args.valB_init,
        valence_lr=args.valence_lr, valence_clip=args.valence_clip,
        # Affect system parameters
        affect_enabled=args.affect_enabled,
        membrane_enabled=args.membrane_enabled,
        brain_membrane_enabled=args.brain_membrane_enabled,
        w_pain=args.w_pain,
        pain_to_temp_gain=args.pain_to_temp_gain,
        pain_semiring_threshold=args.pain_semiring_threshold,
        w_membrane=args.w_membrane,
        membrane_r_min=args.membrane_r_min,
        membrane_r_gain_arousal=args.membrane_r_gain_arousal,
        membrane_r_gain_pain=args.membrane_r_gain_pain,
        affect_rho_v=args.affect_rho_v,
        affect_rho_a=args.affect_rho_a,
        affect_rho_c=args.affect_rho_c,
        affect_rho_p=args.affect_rho_p
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
        max_steps=args.max_steps, seed=args.seed,
        reward_A=args.reward_A, reward_B=args.reward_B
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
    
    # ASCII debug mode
    ascii_parser = subparsers.add_parser("ascii", help="Run episode with ASCII visualization")
    add_common_args(ascii_parser)
    ascii_parser.add_argument("--show-every", type=int, default=10, help="Show state every N steps")
    ascii_parser.add_argument("--show-fields", action="store_true", help="Show field values")
    
    return parser


def run_interactive(args):
    """Run interactive viewer mode."""
    set_global_seed(args.seed)
    
    # Create configs
    env_cfg = EnvConfig(
        H=args.H, W=args.W, win=args.win, p_wall=args.p_wall,
        n_targets_A=args.nA, n_targets_B=args.nB,
        max_steps=args.max_steps, seed=args.seed,
        reward_A=args.reward_A, reward_B=args.reward_B
    )
    
    agent_cfg = AgentConfig(
        seed_strength=args.seed_strength, scent_diff=args.scent_diff,
        scent_decay=args.scent_decay, scent_steps=args.scent_steps,
        v_inj=args.v_inj, v_decay=args.v_decay, v_diff=args.v_diff,
        k_repulse=args.k_repulse, wander=args.wander, stay_thresh=args.stay_thresh,
        anti_stuck_after=args.anti_stuck_after, anti_stuck_temp=args.anti_stuck_temp,
        internal_think=args.internal_think, seed=args.seed,
        # Valence learning
        valA_init=args.valA_init, valB_init=args.valB_init,
        valence_lr=args.valence_lr, valence_clip=args.valence_clip,
        # Affect system parameters
        affect_enabled=args.affect_enabled,
        membrane_enabled=args.membrane_enabled,
        brain_membrane_enabled=args.brain_membrane_enabled,
        w_pain=args.w_pain,
        pain_to_temp_gain=args.pain_to_temp_gain,
        pain_semiring_threshold=args.pain_semiring_threshold,
        w_membrane=args.w_membrane,
        membrane_r_min=args.membrane_r_min,
        membrane_r_gain_arousal=args.membrane_r_gain_arousal,
        membrane_r_gain_pain=args.membrane_r_gain_pain,
        affect_rho_v=args.affect_rho_v,
        affect_rho_a=args.affect_rho_a,
        affect_rho_c=args.affect_rho_c,
        affect_rho_p=args.affect_rho_p
    )
    
    ablate = Ablations(
        trail=args.trail, novelty=args.novelty, 
        corner=args.corner, schema=args.schema
    )
    
    # Create environment and agent
    env = ForageWorld(env_cfg)
    if hasattr(args, 'controller') and args.controller == "field":
        print("[interactive] Using FieldController")
        adapter = ForageAdapter(env)
        agent = FieldController(env, adapter, agent_cfg, ablate)
    else:
        print("[interactive] Using ChemotaxisAgentCA")
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
    if metrics.targets_collected:
        print(f"[interactive] Targets: A={metrics.targets_collected.get('A', 0)}, B={metrics.targets_collected.get('B', 0)}")
    if metrics.valence_snapshot:
        print(f"[interactive] Final valences: {metrics.valence_snapshot}")
    if metrics.mean_cosine:
        print(f"[interactive] Gradient-motion alignment: {metrics.mean_cosine:.3f}")
    
    # Display safety metrics if affect system is enabled
    if hasattr(args, 'affect_enabled') and args.affect_enabled:
        print(f"[interactive] === SAFETY METRICS (Affect Enabled) ===")
        print(f"[interactive] Bumps/100 steps: {metrics.bumps_per_100:.2f}")
        print(f"[interactive] Mean pain: {metrics.mean_pain:.3f}")
        print(f"[interactive] Max pain: {metrics.max_pain:.3f}")
        print(f"[interactive] Mean wall distance: {metrics.mean_wall_distance:.2f}")
        
        # Display final affect state
        if hasattr(metrics, 'affect_history') and metrics.affect_history:
            final_affect = metrics.affect_history[-1]
            print(f"[interactive] Final affect state:")
            print(f"[interactive]   Valence: {final_affect['valence']:.3f}")
            print(f"[interactive]   Arousal: {final_affect['arousal']:.3f}")
            print(f"[interactive]   Control: {final_affect['control']:.3f}")
            print(f"[interactive]   Pain: {final_affect['pain']:.3f}")
    
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


def run_ascii(args):
    """Run ASCII debug mode."""
    set_global_seed(args.seed)
    
    # Create configs
    env_cfg = EnvConfig(
        H=args.H, W=args.W, win=args.win, p_wall=args.p_wall,
        n_targets_A=args.nA, n_targets_B=args.nB,
        max_steps=args.max_steps, seed=args.seed,
        reward_A=args.reward_A, reward_B=args.reward_B
    )
    
    agent_cfg = AgentConfig(
        seed_strength=args.seed_strength, scent_diff=args.scent_diff,
        scent_decay=args.scent_decay, scent_steps=args.scent_steps,
        v_inj=args.v_inj, v_decay=args.v_decay, v_diff=args.v_diff,
        k_repulse=args.k_repulse, wander=args.wander, stay_thresh=args.stay_thresh,
        anti_stuck_after=args.anti_stuck_after, anti_stuck_temp=args.anti_stuck_temp,
        internal_think=args.internal_think, seed=args.seed,
        # Valence learning
        valA_init=args.valA_init, valB_init=args.valB_init,
        valence_lr=args.valence_lr, valence_clip=args.valence_clip,
        # Affect system parameters
        affect_enabled=args.affect_enabled,
        membrane_enabled=args.membrane_enabled,
        brain_membrane_enabled=args.brain_membrane_enabled,
        w_pain=args.w_pain,
        pain_to_temp_gain=args.pain_to_temp_gain,
        pain_semiring_threshold=args.pain_semiring_threshold,
        w_membrane=args.w_membrane,
        membrane_r_min=args.membrane_r_min,
        membrane_r_gain_arousal=args.membrane_r_gain_arousal,
        membrane_r_gain_pain=args.membrane_r_gain_pain,
        affect_rho_v=args.affect_rho_v,
        affect_rho_a=args.affect_rho_a,
        affect_rho_c=args.affect_rho_c,
        affect_rho_p=args.affect_rho_p
    )
    
    ablate = Ablations(
        trail=args.trail, novelty=args.novelty, 
        corner=args.corner, schema=args.schema
    )
    
    # Create environment and agent
    env = ForageWorld(env_cfg)
    if hasattr(args, 'controller') and args.controller == "field":
        print("[interactive] Using FieldController")
        adapter = ForageAdapter(env)
        agent = FieldController(env, adapter, agent_cfg, ablate)
    else:
        print("[interactive] Using ChemotaxisAgentCA")
        agent = ChemotaxisAgentCA(env, agent_cfg, ablate)
    
    # Run with ASCII visualization
    from efi.core import effective_potential, pick_action_from_potential, corner_hazard
    
    obs = env.reset()
    agent.reset()
    walls_mask = env.walls.copy()
    Hc = corner_hazard(walls_mask) if ablate.corner else np.zeros_like(walls_mask)
    
    print(f"[ascii] Starting episode (seed={args.seed})")
    print(f"[ascii] Grid: {args.H}x{args.W}, Targets: A={args.nA}, B={args.nB}")
    print(f"[ascii] Initial position: ({env.y}, {env.x})")
    
    # Show initial state
    print("\n=== Initial World ===")
    for y in range(env.H):
        row = []
        for x in range(env.W):
            if env.walls[y, x]:
                row.append('#')
            elif y == env.y and x == env.x:
                row.append('@')
            elif env.TA[y, x]:
                row.append('A')
            elif env.TB[y, x]:
                row.append('B')
            else:
                row.append('.')
        print(''.join(row))
    
    ep_ret = 0.0
    positions = []
    
    for t in range(args.max_steps):
        _, fields = agent.step(obs)
        GA = fields["GA"]; GB = fields["GB"]
        Vtrail = fields["Vtrail"] if ablate.trail else np.zeros_like(GA)
        Novel = fields["Novel"] if ablate.novelty else np.zeros_like(GA)
        
        # Frontier blending
        U = fields.get("Frontier", np.zeros_like(GA))
        if ablate.novelty:
            trail_here = Vtrail[env.y, env.x]
            frontier_weight = max(0.0, 0.25 * (1.0 - trail_here / 3.0))
            Novel = Novel + frontier_weight * U
        
        P_eff = effective_potential(GA, GB, Novel, Vtrail, Hc,
                                    wA=1.0, wB=0.9, wN=0.7,
                                    kV=0.6, kH=0.5)
        
        # Action selection
        trail_here = Vtrail[env.y, env.x]
        if trail_here > 2.0:
            temp = 0.5 + (trail_here - 2.0) * 0.5
            temp = min(temp, 2.0)
            no_backtrack = True
            momentum = 0.0
        else:
            temp = 0.0
            no_backtrack = False
            momentum = 0.05
        
        a = pick_action_from_potential(
            P_eff, env.y, env.x, walls_mask,
            temperature=temp,
            last_action=getattr(agent, "last_action", None),
            no_backtrack=no_backtrack,
            momentum=momentum
        )
        agent.last_action = a
        
        # Step environment
        old_pos = (env.y, env.x)
        obs, r, done, info = env.step(a)
        ep_ret += r
        new_pos = (env.y, env.x)
        positions.append(new_pos)
        
        # Update stuck counter
        if not info.get("moved", False):
            agent.stuck_count += 1
        else:
            agent.stuck_count = 0
        agent.last_pos = new_pos
        
        # Display every N steps
        if (t + 1) % args.show_every == 0 or done:
            print(f"\n=== Step {t+1} ===")
            print(f"Action: {['↑','↓','←','→'][a]}, Moved: {info.get('moved')}")
            print(f"Position: {old_pos} -> {new_pos}")
            print(f"Trail: {trail_here:.2f}, Temp: {temp:.2f}, Return: {ep_ret:.2f}")
            
            if args.show_fields:
                print(f"GA: {GA[env.y, env.x]:.3f}, GB: {GB[env.y, env.x]:.3f}")
                print(f"Novel: {Novel[env.y, env.x]:.3f}, P_eff: {P_eff[env.y, env.x]:.3f}")
            
            # Show current world state
            for y in range(env.H):
                row = []
                for x in range(env.W):
                    if env.walls[y, x]:
                        row.append('#')
                    elif y == env.y and x == env.x:
                        row.append('@')
                    elif env.TA[y, x]:
                        row.append('A')
                    elif env.TB[y, x]:
                        row.append('B')
                    else:
                        row.append('.')
                print(''.join(row))
        
        if done:
            break
    
    print(f"\n[ascii] Episode complete!")
    print(f"[ascii] Final return: {ep_ret:.2f}")
    print(f"[ascii] Steps: {t+1}")
    
    # Check for oscillation
    if len(positions) >= 10:
        unique_last_10 = len(set(positions[-10:]))
        if unique_last_10 <= 3:
            print(f"[ascii] Warning: Agent was stuck in small area (only {unique_last_10} unique positions in last 10 steps)")


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
    elif args.mode == "ascii":
        run_ascii(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()