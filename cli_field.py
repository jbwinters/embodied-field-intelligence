#!/usr/bin/env python3
"""Enhanced CLI with field controller support."""

import argparse
from pathlib import Path

from efi.configs import EnvConfig, AgentConfig, SchemaConfig, Ablations
from efi.core import set_global_seed, ensure_dir, ts
from efi.envs import ForageWorld
from efi.agents import ChemotaxisAgentCA, SchemaField, FieldController, ForageAdapter
from efi.evaluation import run_episode
from efi.visualization.html_viewer import create_html_viewer


def run_interactive_field(args):
    """Run interactive viewer with field controller."""
    set_global_seed(args.seed)
    
    # Create configs with B as undesirable
    env_cfg = EnvConfig(
        H=args.H, W=args.W, win=args.win, p_wall=args.p_wall,
        n_targets_A=args.nA, n_targets_B=args.nB,
        max_steps=args.max_steps, 
        reward_A=args.reward_A,
        reward_B=args.reward_B,
        seed=args.seed
    )
    
    agent_cfg = AgentConfig(
        valA_init=args.valA_init,
        valB_init=args.valB_init,
        valence_lr=args.valence_lr,
        valence_clip=args.valence_clip,
        seed_strength=args.seed_strength, 
        scent_diff=args.scent_diff,
        scent_decay=args.scent_decay, 
        scent_steps=args.scent_steps,
        v_inj=args.v_inj, 
        v_decay=args.v_decay, 
        v_diff=args.v_diff,
        seed=args.seed
    )
    
    ablate = Ablations(
        trail=args.trail, 
        novelty=args.novelty, 
        corner=args.corner, 
        schema=args.schema
    )
    
    # Create environment
    env = ForageWorld(env_cfg)
    
    # Create agent based on controller type
    if args.controller == "field":
        print("[interactive] Using new FieldController")
        adapter = ForageAdapter(env)
        agent = FieldController(env, adapter, agent_cfg, ablate)
    else:
        print("[interactive] Using ChemotaxisAgentCA")
        agent = ChemotaxisAgentCA(env, agent_cfg, ablate)
    
    # Create schema if enabled
    schema = None
    if args.schema:
        schema_cfg = SchemaConfig(
            enabled=1, 
            tile=args.schema_tile, 
            K=args.schema_K, 
            eta=args.schema_eta,
            slowness=args.schema_slowness, 
            bcm_tau=args.schema_bcm_tau,
            comp_k=args.schema_comp_k, 
            diff=args.schema_diff, 
            decay=args.schema_decay,
            steps=args.schema_steps, 
            alpha_schema=args.schema_alpha, 
            seed=args.seed
        )
        schema = SchemaField(env.H, env.W, 6, schema_cfg)
    
    # Run episode with field recording
    print("[interactive] Running episode and collecting data...")
    print(f"[interactive] Initial valences: {agent.valence if hasattr(agent, 'valence') else {'A': agent.valA, 'B': agent.valB}}")
    
    ret, _, metrics, episode_data = run_episode(
        env, agent, schema, ablate, 
        render="none", 
        record=False,
        record_fields=True
    )
    
    print(f"[interactive] Episode complete. Return: {ret:+.3f}, Steps: {metrics.steps}")
    print(f"[interactive] Targets collected: A={metrics.targets_collected.get('A', 0)}, B={metrics.targets_collected.get('B', 0)}")
    print(f"[interactive] Final valences: {metrics.valence_snapshot}")
    if metrics.mean_cosine:
        print(f"[interactive] Mean gradient-motion alignment: {metrics.mean_cosine:.3f}")
    
    # Create HTML viewer
    if episode_data:
        out_dir = ensure_dir(args.out or "runs")
        html_path = create_html_viewer(episode_data, out_dir / f"interactive_{ts()}.html")
        print(f"[interactive] HTML viewer saved to: {html_path}")
        print(f"[interactive] Open this file in a web browser to interact with the episode")
        
        # Try to open in browser
        try:
            import webbrowser
            webbrowser.open(f"file://{html_path}")
            print("[interactive] Opening in default browser...")
        except:
            pass


def main():
    parser = argparse.ArgumentParser(description="Field Controller Interactive Viewer")
    
    # Controller selection
    parser.add_argument("--controller", type=str, default="chemotaxis", 
                       choices=["chemotaxis", "field"],
                       help="Controller type to use")
    
    # Environment
    parser.add_argument("--H", type=int, default=40, help="Grid height")
    parser.add_argument("--W", type=int, default=40, help="Grid width")
    parser.add_argument("--win", type=int, default=5, help="Observation window size")
    parser.add_argument("--p-wall", type=float, default=0.12, help="Wall probability")
    parser.add_argument("--nA", type=int, default=35, help="Number of A targets")
    parser.add_argument("--nB", type=int, default=55, help="Number of B targets")
    parser.add_argument("--max-steps", type=int, default=600, help="Max episode steps")
    parser.add_argument("--seed", type=int, default=6, help="Random seed")
    
    # Rewards (B is undesirable)
    parser.add_argument("--reward-A", type=float, default=1.0, help="Reward for collecting A")
    parser.add_argument("--reward-B", type=float, default=-0.5, help="Reward for collecting B (negative = undesirable)")
    
    # Valence learning
    parser.add_argument("--valA-init", type=float, default=1.0, help="Initial valence for A")
    parser.add_argument("--valB-init", type=float, default=0.1, help="Initial valence for B (starts positive to discover it's bad)")
    parser.add_argument("--valence-lr", type=float, default=0.25, help="Valence learning rate")
    parser.add_argument("--valence-clip", type=float, default=1.5, help="Max absolute valence value")
    
    # Field dynamics
    parser.add_argument("--seed-strength", type=float, default=1.0)
    parser.add_argument("--scent-diff", type=float, default=0.12)
    parser.add_argument("--scent-decay", type=float, default=0.008)
    parser.add_argument("--scent-steps", type=int, default=3)
    parser.add_argument("--v-inj", type=float, default=1.0)
    parser.add_argument("--v-decay", type=float, default=0.012)
    parser.add_argument("--v-diff", type=float, default=0.08)
    
    # Ablations
    parser.add_argument("--novelty", type=int, default=1, choices=[0,1])
    parser.add_argument("--trail", type=int, default=1, choices=[0,1])
    parser.add_argument("--corner", type=int, default=1, choices=[0,1])
    parser.add_argument("--schema", type=int, default=1, choices=[0,1])
    
    # Schema parameters
    parser.add_argument("--schema-tile", type=int, default=5)
    parser.add_argument("--schema-K", type=int, default=4)
    parser.add_argument("--schema-eta", type=float, default=0.03)
    parser.add_argument("--schema-slowness", type=float, default=0.02)
    parser.add_argument("--schema-bcm-tau", type=float, default=0.01)
    parser.add_argument("--schema-comp-k", type=int, default=1)
    parser.add_argument("--schema-diff", type=float, default=0.08)
    parser.add_argument("--schema-decay", type=float, default=0.005)
    parser.add_argument("--schema-steps", type=int, default=1)
    parser.add_argument("--schema-alpha", type=float, default=0.35)
    
    # Output
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    run_interactive_field(args)


if __name__ == "__main__":
    main()