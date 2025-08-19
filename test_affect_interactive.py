#!/usr/bin/env python3
"""Test Phase 1 affect system with interactive viewer."""

import sys
from efi.configs import EnvConfig, AgentConfig, SchemaConfig, Ablations
from efi.envs import ForageWorld
from efi.agents import ChemotaxisAgentCA, SchemaField, FieldController
from efi.evaluation import run_episode
from efi.visualization import InteractiveViewer


def run_interactive_with_affect():
    """Run interactive viewer with affect system enabled."""
    
    # Your typical configuration
    env_cfg = EnvConfig(
        seed=6,
        H=40,
        W=40,
        n_targets_A=35,
        n_targets_B=55,
        reward_A=1.0,
        reward_B=-0.5,
        max_steps=500  # Longer for larger world
    )
    
    # Agent configuration WITH affect system
    agent_cfg = AgentConfig(
        seed=6,
        valence_lr=0.25,
        valA_init=0.1,
        valB_init=0.1,
        
        # AFFECT SYSTEM ENABLED
        affect_enabled=True,
        w_pain=0.7,                    # Pain field weight
        pain_to_temp_gain=0.6,         # Pain increases temperature
        
        # Membrane settings
        membrane_enabled=True,
        w_membrane=0.6,                # Membrane field weight
        membrane_r_min=1.0,            # Base membrane radius
        membrane_r_gain_arousal=1.0,   # Arousal expands membrane
        membrane_r_gain_pain=1.5,      # Pain expands membrane more
        
        # Brain membrane (learning protection)
        brain_membrane_enabled=True,
        brain_membrane_suppress=0.5,   # Suppress learning under pain
        brain_membrane_min_rate=0.1,   # Never stop learning completely
        
        # Nociception weights (what causes pain)
        pain_bump_weight=0.5,          # Bumping walls
        pain_reward_weight=0.3,        # Negative rewards (B targets)
        pain_prox_weight=0.1,          # Being near walls
        pain_stuck_weight=0.1,         # Being stuck
    )
    
    # Schema configuration (if using field controller)
    schema_cfg = SchemaConfig(enabled=True)
    
    # Ablations
    ablate = Ablations()
    
    # Create environment and agent
    env = ForageWorld(env_cfg)
    
    # Use field controller as you specified
    controller_type = "field"
    if controller_type == "field":
        schema = SchemaField(env, schema_cfg)
        agent = FieldController(env, agent_cfg, schema_cfg, ablate)
    else:
        agent = ChemotaxisAgentCA(env, agent_cfg, ablate)
        schema = SchemaField(env, schema_cfg) if schema_cfg.enabled else None
    
    print("=" * 70)
    print("PHASE 1 AFFECT SYSTEM TEST")
    print("=" * 70)
    print(f"Environment: {env_cfg.H}x{env_cfg.W}, A={env_cfg.n_targets_A}, B={env_cfg.n_targets_B}")
    print(f"Rewards: A={env_cfg.reward_A}, B={env_cfg.reward_B}")
    print(f"Controller: {controller_type}")
    print()
    print("AFFECT FEATURES ENABLED:")
    print(f"  ✓ Pain system (w_pain={agent_cfg.w_pain})")
    print(f"  ✓ Protective membrane (w_membrane={agent_cfg.w_membrane})")
    print(f"  ✓ Brain membrane (learning protection)")
    print(f"  ✓ Dynamic temperature (pain→temp gain={agent_cfg.pain_to_temp_gain})")
    print()
    print("SAFETY METRICS TRACKED:")
    print("  • Bumps per 100 steps")
    print("  • Mean/max pain levels")
    print("  • Average wall distance")
    print("  • Affect state (valence, arousal, control)")
    print("=" * 70)
    print()
    print("Running episode and collecting data...")
    
    # Run episode with field recording
    return_val, frames, metrics, episode_data = run_episode(
        env, agent, schema, ablate,
        render="none",
        record=True,
        record_fields=True
    )
    
    print(f"\nEpisode complete!")
    print(f"  Return: {metrics.total_return:.2f}")
    print(f"  Steps: {metrics.steps}")
    print(f"  A collected: {metrics.targets_collected.get('A', 0)}")
    print(f"  B collected: {metrics.targets_collected.get('B', 0)}")
    print(f"  Bumps/100: {metrics.bumps_per_100:.2f}")
    print(f"  Mean pain: {metrics.mean_pain:.3f}")
    print(f"  Max pain: {metrics.max_pain:.3f}")
    print(f"  Mean wall distance: {metrics.mean_wall_distance:.2f}")
    
    if metrics.affect_history:
        final = metrics.affect_history[-1]
        print(f"\nFinal affect state:")
        print(f"  Valence: {final['valence']:.3f}")
        print(f"  Arousal: {final['arousal']:.3f}")
        print(f"  Control: {final['control']:.3f}")
        print(f"  Pain: {final['pain']:.3f}")
    
    print("\nLaunching interactive viewer...")
    print("Controls:")
    print("  SPACE: Play/pause")
    print("  LEFT/RIGHT: Step through frames")
    print("  UP/DOWN: Speed control")
    print("  Click field names to toggle visibility")
    print()
    
    # Launch interactive viewer
    if episode_data:
        # Add affect visualization to fields
        if metrics.affect_history:
            # Add pain field visualization
            for i, frame_data in enumerate(episode_data['fields']):
                if i < len(metrics.affect_history):
                    affect = metrics.affect_history[i]
                    # Create a pain visualization field
                    H, W = frame_data['GA'].shape
                    pain_viz = np.zeros((H, W), dtype=np.float32)
                    # Show pain level as a gradient
                    pain_viz[:] = affect['pain']
                    frame_data['Pain'] = pain_viz
                    
                    # Update info with affect data
                    frame_data['info']['pain'] = affect['pain']
                    frame_data['info']['arousal'] = affect['arousal']
                    frame_data['info']['valence'] = affect['valence']
                    frame_data['info']['control'] = affect['control']
        
        viewer = InteractiveViewer(episode_data)
        viewer.show()
    else:
        print("No field data recorded for visualization")


if __name__ == "__main__":
    run_interactive_with_affect()