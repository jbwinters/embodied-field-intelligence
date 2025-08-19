#!/usr/bin/env python3
"""Test the improvements with focused experiments."""

import numpy as np
from dataclasses import replace

from efi.configs import EnvConfig, AgentConfig, SchemaConfig, Ablations
from efi.envs import ForageWorld
from efi.agents import ChemotaxisAgentCA, SchemaField
from efi.evaluation import run_episode
from efi.core import set_global_seed


def ascii_render(env, agent, fields=None):
    """Simple ASCII visualization of the environment and agent state."""
    H, W = env.H, env.W
    
    # Create display grid
    display = [['.' for _ in range(W)] for _ in range(H)]
    
    # Add walls
    for y in range(H):
        for x in range(W):
            if env.walls[y, x]:
                display[y][x] = '#'
    
    # Add targets
    for y in range(H):
        for x in range(W):
            if env.TA[y, x]:
                display[y][x] = 'A'
            elif env.TB[y, x]:
                display[y][x] = 'B'
    
    # Add agent
    display[env.y][env.x] = '@'
    
    # Print grid
    print("\n" + "="*W)
    for row in display:
        print(''.join(row))
    
    # Print key field strengths at agent position if available
    if fields:
        y, x = env.y, env.x
        print(f"\nAt agent pos ({y},{x}):")
        print(f"  GA: {fields['GA'][y,x]:.3f}")
        print(f"  GB: {fields['GB'][y,x]:.3f}")
        print(f"  Trail: {fields['Vtrail'][y,x]:.3f}")
        print(f"  Novel: {fields['Novel'][y,x]:.3f}")
        if 'WallProx' in fields:
            print(f"  WallProx: {fields['WallProx'][y,x]:.3f}")


def test_wall_proximity():
    """Test A4: Wall proximity reduces wall-hugging in corridors."""
    print("\n" + "="*60)
    print("TEST 1: Wall Proximity in Narrow Corridor")
    print("="*60)
    
    # Create a corridor environment
    cfg = EnvConfig(H=10, W=20, p_wall=0.0, n_targets_A=1, n_targets_B=0, max_steps=50)
    
    for use_wall_prox in [False, True]:
        print(f"\n{'WITH' if use_wall_prox else 'WITHOUT'} Wall Proximity Field:")
        
        env = ForageWorld(cfg)
        obs = env.reset()  # Need to reset first
        # Create corridor manually after reset
        env.walls[:] = True
        env.walls[4:6, :] = False  # 2-cell wide corridor
        env.TA[:] = False
        env.TA[4, 18] = True  # Target at end
        env.y, env.x = 4, 1  # Start position
        
        agent_cfg = AgentConfig(seed=42)
        if use_wall_prox:
            agent_cfg.w_wall_prox = 0.5  # Strong wall repulsion
        
        agent = ChemotaxisAgentCA(env, agent_cfg, Ablations(wall_proximity=use_wall_prox))
        agent.reset()
        bumps = 0
        
        for step in range(20):
            obs_vec = env._obs()
            _, fields = agent.step(obs_vec)
            
            # Show state every 5 steps
            if step % 5 == 0:
                print(f"\nStep {step}:")
                ascii_render(env, agent, fields)
            
            # Choose action (simplified from runner)
            from efi.core import pick_action_from_potential, corner_hazard
            walls_mask = env.walls.copy()
            
            # Compose potential (simplified)
            P_eff = agent.compose_P(
                walls_mask=walls_mask,
                corner_field=corner_hazard(walls_mask),
                wall_prox_field=fields.get('WallProx'),
                schema_bias=None,
                frontier_weight=0.1
            )
            
            a = pick_action_from_potential(P_eff, env.y, env.x, walls_mask)
            obs, r, done, info = env.step(a)
            
            if not info.get('moved', True):
                bumps += 1
            
            if done:
                break
        
        print(f"\nTotal bumps: {bumps}")
        print(f"Final position: ({env.y}, {env.x})")


def test_aversive_learning():
    """Test A1/A2: Schema learns to avoid B targets through experience."""
    print("\n" + "="*60)
    print("TEST 2: Aversive Schema Learning")
    print("="*60)
    
    cfg = EnvConfig(
        H=15, W=15, 
        n_targets_A=2, n_targets_B=4,
        reward_A=1.0, reward_B=-0.5,  # B is undesirable
        max_steps=100,
        p_wall=0.1
    )
    
    env = ForageWorld(cfg)
    agent_cfg = AgentConfig(
        valB_init=0.5,  # Start thinking B is somewhat good
        valence_lr=0.3  # Learn quickly
    )
    # These are accessed via getattr, not constructor params
    agent_cfg.valence_lr_step = 0.005  # Counterfactual learning rate
    agent_cfg.valence_step_baseline = cfg.step_cost  # Baseline for counterfactual
    
    agent = ChemotaxisAgentCA(env, agent_cfg, Ablations())
    schema = SchemaField(env.H, env.W, feature_dim=6, 
                        cfg=SchemaConfig(K=4, enabled=True))
    
    print("\nInitial valences:")
    print(f"  A: {agent.valence['A']:.3f}")
    print(f"  B: {agent.valence['B']:.3f}")
    
    # Run episode
    total_return, frames, metrics, field_history = run_episode(
        env, agent, schema, Ablations(), record_fields=True
    )
    
    print(f"\nEpisode complete:")
    print(f"  Total return: {total_return:.2f}")
    print(f"  A collected: {metrics.targets_collected['A']}")
    print(f"  B collected: {metrics.targets_collected['B']}")
    
    print(f"\nFinal valences (after learning):")
    print(f"  A: {agent.valence['A']:.3f}")
    print(f"  B: {agent.valence['B']:.3f}")
    
    # Check if any schema prototypes learned negative valence
    negative_tiles = np.sum(schema.q < -0.1)
    positive_tiles = np.sum(schema.q > 0.1)
    print(f"\nSchema valences:")
    print(f"  Positive prototypes: {positive_tiles}")
    print(f"  Negative prototypes: {negative_tiles}")
    
    # Show final state
    print("\nFinal environment state:")
    ascii_render(env, agent, field_history[-1] if field_history else None)


def test_field_flatness_exploration():
    """Test A3: Temperature increases in flat field regions."""
    print("\n" + "="*60)
    print("TEST 3: Exploration in Flat Field Regions")
    print("="*60)
    
    # Create sparse environment where agent might get stuck in flat regions
    cfg = EnvConfig(
        H=20, W=20, 
        n_targets_A=1, n_targets_B=0,  # Single distant target
        p_wall=0.15,
        max_steps=50
    )
    
    env = ForageWorld(cfg)
    # Place target far from start
    env.TA[:] = False
    env.TA[18, 18] = True
    env.y, env.x = 1, 1  # Start far away
    
    agent = ChemotaxisAgentCA(env, AgentConfig(seed=42), Ablations())
    
    obs = env.reset()
    agent.reset()
    
    print("Tracking exploration with gradient-based temperature...")
    
    for step in range(30):
        obs_vec = env._obs()
        _, fields = agent.step(obs_vec)
        
        # Compute gradient magnitude for temperature
        from efi.core import corner_hazard
        P_eff = agent.compose_P(
            walls_mask=env.walls,
            corner_field=corner_hazard(env.walls),
            wall_prox_field=fields.get('WallProx'),
            schema_bias=None,
            frontier_weight=0.2
        )
        
        gy, gx = np.gradient(P_eff.astype(np.float32))
        grad_mag = np.sqrt(gy[env.y, env.x]**2 + gx[env.y, env.x]**2)
        
        # Temperature from flatness
        epsilon = 0.01
        alpha_grad = 0.3
        temp_flatness = alpha_grad / (epsilon + grad_mag)
        
        if step % 10 == 0:
            print(f"\nStep {step}:")
            print(f"  Position: ({env.y}, {env.x})")
            print(f"  Gradient magnitude: {grad_mag:.4f}")
            print(f"  Temperature from flatness: {temp_flatness:.3f}")
            print(f"  GA field strength: {fields['GA'][env.y, env.x]:.4f}")
            
            # Simple ASCII view
            if step == 0:
                ascii_render(env, agent)
        
        # Take action
        from efi.core import pick_action_from_potential
        a = pick_action_from_potential(
            P_eff, env.y, env.x, env.walls,
            temperature=temp_flatness  # Use flatness-based temperature
        )
        
        obs, r, done, info = env.step(a)
        if done:
            print(f"\nTarget reached at step {step}!")
            break


def test_semiring_modes():
    """Test B1: Different aggregation modes change behavior."""
    print("\n" + "="*60)
    print("TEST 4: Semiring Aggregation Modes")
    print("="*60)
    
    # Environment with multiple conflicting signals
    cfg = EnvConfig(H=12, W=12, n_targets_A=3, n_targets_B=3, p_wall=0.05)
    
    modes = [
        ("linear", "linear", "Linear aggregation"),
        ("lse", "linear", "LSE for attractors (emphasize strong)"),
        ("linear", "maxplus", "Max-plus for repulsors (strongest hazard)"),
        ("maxplus", "maxplus", "Max-plus both (winner takes all)")
    ]
    
    for mode_attr, mode_rep, desc in modes:
        print(f"\n{desc}:")
        print(f"  Attractors: {mode_attr}, Repulsors: {mode_rep}")
        
        env = ForageWorld(cfg)
        agent_cfg = AgentConfig(seed=42)
        agent = ChemotaxisAgentCA(env, agent_cfg, Ablations())
        
        obs = env.reset()
        agent.reset()
        
        # Run a few steps
        for step in range(5):
            obs_vec = env._obs()
            _, fields = agent.step(obs_vec)
            
            # Compose with specific mode
            from efi.core import compose_potential
            attractors = {"A": fields['GA'], "B": fields['GB']}
            repulsors = {"Trail": fields['Vtrail']}
            w_attr = {"A": 1.0, "B": 0.8}
            w_rep = {"Trail": 0.5}
            
            P_eff = compose_potential(
                attractors, repulsors, w_attr, w_rep,
                mode=mode_attr,  # Default mode
                mode_attr=mode_attr,
                mode_rep=mode_rep,
                beta_attr=2.0,
                beta_rep=2.0
            )
            
            if step == 0:
                # Sample some points to show aggregation differences
                y, x = env.y, env.x
                print(f"  At agent ({y},{x}):")
                print(f"    GA={fields['GA'][y,x]:.3f}, GB={fields['GB'][y,x]:.3f}")
                print(f"    P_eff={P_eff[y,x]:.3f}")
            
            from efi.core import pick_action_from_potential
            a = pick_action_from_potential(P_eff, env.y, env.x, env.walls)
            obs, r, done, info = env.step(a)
            
            if done:
                break
        
        print(f"  Moved to: ({env.y}, {env.x})")


if __name__ == "__main__":
    set_global_seed(42)
    
    # Run all tests
    test_wall_proximity()
    test_aversive_learning()
    test_field_flatness_exploration()
    test_semiring_modes()
    
    print("\n" + "="*60)
    print("All experiments complete!")
    print("="*60)