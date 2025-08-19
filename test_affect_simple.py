#!/usr/bin/env python3
"""Simple visual test of affect system dynamics."""

import numpy as np
from efi.configs import EnvConfig, AgentConfig, Ablations
from efi.envs import ForageWorld
from efi.agents import ChemotaxisAgentCA
from efi.core import (
    AffectState, compute_nociception, update_affect,
    wall_proximity_field, peripersonal_field
)


def visualize_simple_scenario():
    """Run a simple scenario and show affect dynamics step by step."""
    
    # Simple 10x10 environment
    env_cfg = EnvConfig(
        H=10, W=10,
        p_wall=0.0,  # Start with no random walls
        n_targets_A=1,
        n_targets_B=1,
        reward_A=1.0,
        reward_B=-1.0,
        max_steps=30,
        seed=999
    )
    
    # Create environment and add some walls manually for predictable layout
    env = ForageWorld(env_cfg)
    env.walls[5, 3:7] = True  # Horizontal wall
    env.walls[3:7, 5] = True  # Vertical wall (creates a cross)
    
    # Agent with affect
    agent_cfg = AgentConfig(
        affect_enabled=True,
        membrane_enabled=True,
        w_pain=0.8,
        w_membrane=0.7,
        pain_to_temp_gain=0.7,
        seed=999
    )
    
    ablate = Ablations()
    
    # Create agent
    agent = ChemotaxisAgentCA(env, agent_cfg, ablate)
    
    print("="*60)
    print("SIMPLE AFFECT DYNAMICS VISUALIZATION")
    print("="*60)
    print("\nEnvironment: 10x10 with cross-shaped wall")
    print("Symbols: @ = agent, # = wall, A = good target, B = bad target")
    print("="*60)
    
    # Initialize
    obs = env.reset()
    agent.reset()
    affect_state = AffectState()
    
    # Run for a few steps manually to show dynamics
    for step in range(15):
        # Get agent's position
        y, x = env.y, env.x
        
        # Compute wall proximity
        W_prox = wall_proximity_field(env.walls, radius=1.5)
        wall_prox_here = W_prox[y, x]
        
        # Simple action (just move randomly for demo)
        action = np.random.randint(4)
        old_pos = (y, x)
        obs, reward, done, info = env.step(action)
        new_pos = (env.y, env.x)
        
        # Compute nociception
        bump = not info.get("moved", False)
        stuck = bump  # Simplified
        neg_reward = min(0, reward)
        
        nociception = compute_nociception(
            bump=bump,
            neg_reward=neg_reward,
            wall_prox_here=wall_prox_here,
            stuck_count=1 if stuck else 0,
            bump_weight=0.5,
            reward_weight=0.3,
            prox_weight=0.1,
            stuck_weight=0.1
        )
        
        # Update affect
        affect_state = update_affect(
            affect_state,
            nociception=nociception,
            surprise=0.0,
            reward=reward,
            rho_v=0.1,
            rho_a=0.1,
            rho_c=0.1,
            rho_p=0.2
        )
        
        # Display every few steps
        if step % 3 == 0 or bump or reward != 0:
            print(f"\n--- Step {step+1} ---")
            print(f"Action: {['↑','↓','←','→'][action]}")
            print(f"Position: {old_pos} → {new_pos}")
            print(f"Moved: {info.get('moved')}, Bump: {bump}")
            print(f"Reward: {reward:.2f}")
            print(f"Wall proximity: {wall_prox_here:.2f}")
            print(f"Nociception: {nociception:.3f}")
            print(f"AFFECT STATE:")
            print(f"  Pain: {affect_state.pain:.3f}")
            print(f"  Valence: {affect_state.valence:+.3f}")
            print(f"  Arousal: {affect_state.arousal:.3f}")
            print(f"  Control: {affect_state.control:.3f}")
            
            # Show grid
            print("\nWorld state:")
            for gy in range(env.H):
                row = []
                for gx in range(env.W):
                    if env.walls[gy, gx]:
                        row.append('#')
                    elif gy == env.y and gx == env.x:
                        row.append('@')
                    elif env.TA[gy, gx]:
                        row.append('A')
                    elif env.TB[gy, gx]:
                        row.append('B')
                    else:
                        # Show membrane field strength
                        if agent_cfg.membrane_enabled:
                            membrane = peripersonal_field(
                                env.walls,
                                np.ones_like(env.walls),  # All visible
                                env.y, env.x,
                                R_base=1.5,
                                arousal=affect_state.arousal,
                                pain=affect_state.pain,
                                R_gain_arousal=1.0,
                                R_gain_pain=1.5
                            )
                            if membrane[gy, gx] > 0.5:
                                row.append('o')  # Strong membrane
                            elif membrane[gy, gx] > 0.1:
                                row.append('.')  # Weak membrane
                            else:
                                row.append(' ')  # No membrane
                        else:
                            row.append('.')
                print(''.join(row))
            
            if bump:
                print("  ⚠️  BUMP! Pain increased")
            if reward > 0:
                print("  ✓ Positive reward! Valence improved")
            elif reward < 0:
                print("  ✗ Negative reward! Valence decreased")
        
        if done:
            break
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("The affect system responds to:")
    print("• Bumps → Immediate pain spike")
    print("• Negative rewards → Pain and negative valence")
    print("• Wall proximity → Mild discomfort")
    print("• Being stuck → Increasing pain over time")
    print("\nPain causes:")
    print("• Higher temperature (more random actions)")
    print("• Expanded membrane (larger safety buffer)")
    print("• Suppressed learning (brain membrane)")


if __name__ == "__main__":
    visualize_simple_scenario()