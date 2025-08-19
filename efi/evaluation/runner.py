"""Episode and experiment runners."""

from typing import List, Optional, Tuple
from dataclasses import asdict

import numpy as np

from ..configs import Ablations, EnvConfig, AgentConfig, SchemaConfig
from ..envs import ForageWorld
from ..agents import ChemotaxisAgentCA, SchemaField, build_features_for_schema
from ..core import (
    corner_hazard,
    wall_proximity_field,
    effective_potential, 
    pick_action_from_potential,
    set_global_seed,
    compose_potential,
    # Affect system
    AffectState,
    compute_nociception,
    update_affect,
    pain_to_temperature,
    pain_field,
    # Membrane system
    peripersonal_field,
    brain_membrane_gate,
    compute_membrane_potential
)
from .metrics import EpisodeMetrics, ExperimentResults


def run_episode(
    env: ForageWorld,
    agent: ChemotaxisAgentCA,
    schema: Optional[SchemaField],
    ablate: Ablations,
    render: str = "none",
    record: bool = False,
    record_fields: bool = False
) -> Tuple[float, List[np.ndarray], EpisodeMetrics, Optional[dict]]:
    obs = env.reset()
    agent.reset()
    walls_mask = env.walls.copy()
    Hc = corner_hazard(walls_mask) if ablate.corner else np.zeros_like(walls_mask, dtype=np.float32)
    
    # Add wall proximity field as additional repulsor
    W_prox = wall_proximity_field(walls_mask, radius=1.5) if getattr(ablate, 'wall_proximity', True) else np.zeros_like(walls_mask, dtype=np.float32)
    
    # Initialize affect state if enabled
    affect_state = AffectState() if agent.cfg.affect_enabled else None
    affect_history = []  # Track affect over time for metrics
    bumps_total = 0
    pain_history = []
    wall_distances = []

    ep_ret = 0.0
    frames, field_frames, world_frames = [], [], []
    steps = 0
    targets_collected = {"A": 0, "B": 0}
    cosines = []  # Track gradient-motion alignment
    prev_yx = (env.y, env.x)

    for t in range(env.max_steps):
        _, fields = agent.step(obs)
        GA = fields["GA"]; GB = fields["GB"]
        Vtrail = fields["Vtrail"] if ablate.trail else np.zeros_like(GA)
        Novel  = fields["Novel"]  if ablate.novelty else np.zeros_like(GA)

        # Compute affect fields if enabled
        pain_field_array = np.zeros_like(GA)
        membrane_field_array = np.zeros_like(GA)
        
        if affect_state and agent.cfg.affect_enabled:
            # Compute membrane field
            membrane_field_array = peripersonal_field(
                agent.known_walls,
                agent.seen,
                env.y,
                env.x,
                agent.cfg.membrane_r_min,
                affect_state.arousal,
                affect_state.pain,
                agent.cfg.membrane_r_gain_arousal,
                agent.cfg.membrane_r_gain_pain
            ) if agent.cfg.membrane_enabled else np.zeros_like(GA)
            
            # Compute pain field
            pain_field_array = pain_field(
                affect_state.pain,
                env.y,
                env.x,
                env.H,
                env.W
            )
        
        # Enhanced frontier blending with trail AND uncertainty
        U = fields.get("Frontier", np.zeros_like(GA))
        trail_here = Vtrail[env.y, env.x]
        
        # Compute uncertainty at current position (1 - seen)
        seen = getattr(agent, 'seen', np.ones_like(GA))
        uncertainty_here = 1.0 - seen[env.y, env.x]
        
        frontier_weight = 0.0
        if ablate.novelty:
            # Base weight modulated by both trail and uncertainty
            lambda_frontier = 0.25  # Base frontier weight
            trail_factor = max(0.0, 1.0 - trail_here / 3.0)  # Reduce when stuck
            uncertainty_factor = uncertainty_here  # Boost in unexplored areas
            
            frontier_weight = lambda_frontier * trail_factor * uncertainty_factor
            frontier_weight = np.clip(frontier_weight, 0.0, lambda_frontier)
            
            # Blend frontier into novelty
            Novel = Novel + frontier_weight * U
        
        # --- Potential composition ---
        if hasattr(agent, "compose_P"):
            # Agent has compose_P method - delegate to it
            # Pass affect state to agent for semiring mode selection
            if affect_state is not None:
                agent.affect_state = affect_state
            
            # First compute schema if needed
            schema_bias = None
            Ssum = np.zeros_like(GA)
            if schema and schema.cfg.enabled and ablate.schema:
                # Need a base potential for schema learning
                # Quick compose without schema to get P_base
                P_base_for_schema = agent.compose_P(
                    walls_mask=walls_mask,
                    corner_field=Hc if ablate.corner else None,
                    wall_prox_field=W_prox,
                    schema_bias=None,
                    frontier_weight=frontier_weight,
                    pain_field=pain_field_array if agent.cfg.affect_enabled and affect_state else None,
                    membrane_field=membrane_field_array if agent.cfg.membrane_enabled and affect_state else None
                )
                feats = build_features_for_schema(GA, GB, Novel, Vtrail, Hc, P_base_for_schema)
                schema.update(feats)
                Ssum = np.sum(schema.Smaps, axis=0).astype(np.float32)
                schema_bias = schema.bias_field()
            
            # Final composition with schema
            P_eff = agent.compose_P(
                walls_mask=walls_mask,
                corner_field=Hc if ablate.corner else None,
                wall_prox_field=W_prox,
                schema_bias=schema_bias,
                frontier_weight=frontier_weight,
                pain_field=pain_field_array if agent.cfg.affect_enabled and affect_state else None,
                membrane_field=membrane_field_array if agent.cfg.membrane_enabled and affect_state else None
            )
        else:
            # Legacy path - manual composition
            # Blend frontier into novelty (as before)
            if ablate.novelty and frontier_weight != 0.0:
                Novel = Novel + frontier_weight * U
            
            if hasattr(agent, 'valence') and isinstance(agent.valence, dict):
                # Channel-agnostic composition
                attractors = {"A": GA, "B": GB, "Novel": Novel}
                repulsors = {
                    "Trail": Vtrail, 
                    "Corner": Hc, 
                    "WallProx": W_prox,
                    "Pain": pain_field_array,
                    "Membrane": membrane_field_array
                }
                w_attr = {
                    "A": agent.valence.get("A", 1.0),
                    "B": agent.valence.get("B", 1.0),
                    "Novel": agent.valence.get("Novel", agent.cfg.w_novel)
                }
                w_rep = {
                    "Trail": agent.cfg.w_trail,
                    "Corner": agent.cfg.w_corner,
                    "WallProx": getattr(agent.cfg, "w_wall_prox", 0.3),  # Default weight for wall proximity
                    "Pain": agent.cfg.w_pain if affect_state else 0.0,
                    "Membrane": agent.cfg.w_membrane if affect_state else 0.0
                }
                # Determine semiring mode based on pain
                mode = "linear"  # default
                if affect_state is not None and affect_state.pain > getattr(agent.cfg, 'pain_semiring_threshold', 0.6):
                    mode = "maxplus"  # Use max-plus semiring under high pain
                
                P_base = compose_potential(attractors, repulsors, w_attr, w_rep, bias=None, mode=mode)
            else:
                # Legacy method for backwards compatibility
                P_base = effective_potential(GA, GB, Novel, Vtrail, Hc,
                                            wA=agent.valA,
                                            wB=agent.valB,
                                            wN=agent.cfg.w_novel,
                                            kV=agent.cfg.w_trail,
                                            kH=agent.cfg.w_corner)

            # Schema bias (learned from base potential, applied after)
            Ssum = np.zeros_like(P_base)
            schema_bias = None
            if schema and schema.cfg.enabled and ablate.schema:
                feats = build_features_for_schema(GA, GB, Novel, Vtrail, Hc, P_base)
                schema.update(feats)
                Ssum = np.sum(schema.Smaps, axis=0).astype(np.float32)
                schema_bias = schema.bias_field()
            
            # Final potential = base + (optional) schema bias
            P_eff = P_base + (schema_bias if schema_bias is not None else 0.0)

        # Enhanced temperature schedule using trail strength AND field flatness
        trail_here = Vtrail[env.y, env.x]
        
        # Compute local gradient magnitude
        gy, gx = np.gradient(P_eff.astype(np.float32))
        grad_mag = np.sqrt(gy[env.y, env.x]**2 + gx[env.y, env.x]**2)
        
        # Temperature from trail (stuck signal)
        if trail_here > 2.0:  # We're oscillating/stuck
            temp_trail = 0.5 + (trail_here - 2.0) * 0.5
            temp_trail = min(temp_trail, 2.0)  # Cap at 2.0
            no_backtrack = True
            momentum = 0.0
        else:
            temp_trail = 0.0
            no_backtrack = False
            momentum = 0.05
        
        # Temperature from field flatness (inverse gradient magnitude)
        epsilon = 0.01  # Small constant to avoid division by zero
        alpha_grad = 0.3  # Weight for gradient-based temperature
        temp_flatness = alpha_grad / (epsilon + grad_mag)
        temp_flatness = min(temp_flatness, 1.0)  # Cap contribution
        
        # Combined temperature (no cap here, will cap after pain boost)
        temp = temp_trail + temp_flatness
        
        # Apply pain-based temperature boost if affect enabled
        if affect_state and agent.cfg.affect_enabled:
            temp = pain_to_temperature(
                temp,
                affect_state.pain,
                affect_state.arousal,
                agent.cfg.pain_to_temp_gain,
                max_temp=3.0  # Unified higher cap to allow pain boost
            )
        else:
            # Cap temperature if no pain boost
            temp = min(temp, 2.5)
        
        a = pick_action_from_potential(
            P_eff, env.y, env.x, walls_mask,
            temperature=temp,
            last_action=getattr(agent, "last_action", None),
            no_backtrack=no_backtrack,
            momentum=momentum
        )

        # Step env
        obs, r, done, info = env.step(a)
        
        # Counterfactual valence learning (every step)
        if hasattr(agent, 'learn_valence_counterfactual'):
            # Compute field values at chosen action and alternatives
            y_prev, x_prev = prev_yx
            displacements = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
            
            # Get field values at the chosen action's destination
            dy_chosen, dx_chosen = displacements[a]
            y_chosen, x_chosen = y_prev + dy_chosen, x_prev + dx_chosen
            if 0 <= y_chosen < env.H and 0 <= x_chosen < env.W and not walls_mask[y_chosen, x_chosen]:
                field_at_action = {
                    "A": GA[y_chosen, x_chosen],
                    "B": GB[y_chosen, x_chosen],
                    "Novel": Novel[y_chosen, x_chosen]
                }
            else:
                field_at_action = {"A": 0.0, "B": 0.0, "Novel": 0.0}
            
            # Get average field values at alternative actions
            alternatives = []
            for i, (dy_alt, dx_alt) in enumerate(displacements):
                if i != a:  # Skip the chosen action
                    y_alt, x_alt = y_prev + dy_alt, x_prev + dx_alt
                    if 0 <= y_alt < env.H and 0 <= x_alt < env.W and not walls_mask[y_alt, x_alt]:
                        alternatives.append({
                            "A": GA[y_alt, x_alt],
                            "B": GB[y_alt, x_alt],
                            "Novel": Novel[y_alt, x_alt]
                        })
            
            if alternatives:
                field_alternatives = {
                    "A": np.mean([alt["A"] for alt in alternatives]),
                    "B": np.mean([alt["B"] for alt in alternatives]),
                    "Novel": np.mean([alt["Novel"] for alt in alternatives])
                }
                # Baseline reward to avoid bias from step_cost; gate on movement
                baseline = float(getattr(agent.cfg, "valence_step_baseline", env.cfg.step_cost))
                r_eff = r - baseline
                if info.get("moved", True):
                    agent.learn_valence_counterfactual(field_at_action, field_alternatives, r_eff)
        
        # Compute gradient-motion alignment (after step) - reuse gradient from temperature calc
        dy, dx = env.y - prev_yx[0], env.x - prev_yx[1]
        if dy != 0 or dx != 0:  # Only if we moved
            # We already computed gy, gx above for temperature schedule
            gvec = np.array([gy[prev_yx[0], prev_yx[1]], gx[prev_yx[0], prev_yx[1]]], dtype=np.float32)
            dvec = np.array([dy, dx], dtype=np.float32)
            if np.linalg.norm(gvec) > 1e-6:
                cos = float(np.dot(gvec, dvec) / (np.linalg.norm(gvec) * np.linalg.norm(dvec)))
                cosines.append(cos)
        prev_yx = (env.y, env.x)
        ep_ret += r
        steps += 1
        agent.last_action = a  # Store for momentum/no-backtrack

        # Update stuck counter from the truth on the ground
        bump = not info.get("moved", False)
        if bump:
            agent.stuck_count += 1
            bumps_total += 1
        else:
            agent.stuck_count = 0
        agent.last_pos = (env.y, env.x)
        
        # Update affect state if enabled
        if affect_state and agent.cfg.affect_enabled:
            # Compute nociception
            wall_prox_here = W_prox[env.y, env.x] if W_prox is not None else 0.0
            neg_reward = min(0, r)  # Only negative part of reward
            
            nociception = compute_nociception(
                bump,
                neg_reward,
                wall_prox_here,
                agent.stuck_count,
                agent.cfg.pain_bump_weight,
                agent.cfg.pain_reward_weight,
                agent.cfg.pain_prox_weight,
                agent.cfg.pain_stuck_weight
            )
            
            # Compute surprise (based on novelty)
            surprise = Novel[env.y, env.x] if Novel is not None else 0.0
            
            # Update affect
            affect_state = update_affect(
                affect_state,
                nociception,
                surprise,
                r,
                agent.cfg.affect_rho_v,
                agent.cfg.affect_rho_a,
                agent.cfg.affect_rho_c,
                agent.cfg.affect_rho_p
            )
            
            # Track for metrics
            pain_history.append(affect_state.pain)
            affect_history.append(affect_state.to_dict())
            
            # Track wall distance
            if agent.known_walls.any():
                from scipy.ndimage import distance_transform_edt
                dist_map = distance_transform_edt(~agent.known_walls)
                wall_distances.append(dist_map[env.y, env.x])

        # Learn from pickups (with brain membrane gating if enabled)
        picked = info.get("picked")
        learning_gate = 1.0
        if affect_state and agent.cfg.brain_membrane_enabled:
            learning_gate = brain_membrane_gate(
                affect_state.pain,
                1.0,
                agent.cfg.brain_membrane_suppress,
                agent.cfg.brain_membrane_min_rate
            )
        
        if picked == "A":
            targets_collected["A"] += 1
            # Apply gated learning
            orig_lr = agent.cfg.valence_lr
            agent.cfg.valence_lr = orig_lr * learning_gate
            agent.learn_valence("A", env.cfg.reward_A)
            agent.cfg.valence_lr = orig_lr
            # Update schema valence with positive reward
            if schema and schema.cfg.enabled:
                schema.update_valence(env.cfg.reward_A * learning_gate)
        elif picked == "B":
            targets_collected["B"] += 1
            # Apply gated learning  
            orig_lr = agent.cfg.valence_lr
            agent.cfg.valence_lr = orig_lr * learning_gate
            agent.learn_valence("B", env.cfg.reward_B)
            agent.cfg.valence_lr = orig_lr
            # Update schema valence with negative reward
            if schema and schema.cfg.enabled:
                schema.update_valence(env.cfg.reward_B * learning_gate)
        
        # Recompute affect fields after state update for accurate visualization
        if affect_state and agent.cfg.affect_enabled and record_fields:
            # Update pain field with current pain level
            pain_field_array = pain_field(
                affect_state.pain,
                env.y,
                env.x,
                env.H, env.W,
                radius=2.0
            )
            
            # Update membrane field with current affect state
            if agent.cfg.membrane_enabled:
                membrane_field_array = peripersonal_field(
                    agent.known_walls,
                    agent.seen,
                    env.y,
                    env.x,
                    agent.cfg.membrane_r_min,
                    arousal=affect_state.arousal,
                    pain=affect_state.pain,
                    R_gain_arousal=agent.cfg.membrane_r_gain_arousal,
                    R_gain_pain=agent.cfg.membrane_r_gain_pain
                )

        # Optionally record frames & fields
        if record:
            world_rgb = env.render_rgb()
            frames.append(world_rgb.copy())

        if record_fields:
            world_rgb = env.render_rgb()
            world_frames.append(world_rgb.copy())
            fields_dict = {
                'GA': GA.copy(),
                'GB': GB.copy(),
                'P_eff': P_eff.copy(),
                'Vtrail': Vtrail.copy(),
                'Novel': Novel.copy(),
                'Ssum': Ssum.copy(),
                'info': {
                    'step': t,
                    'return': ep_ret,
                    'action': a,
                    'stuck_count': agent.stuck_count,
                    'reward': r,
                    'valA': float(agent.valA),
                    'valB': float(agent.valB)
                }
            }
            
            # Add affect fields if enabled
            if affect_state and agent.cfg.affect_enabled:
                fields_dict['Pain'] = pain_field_array.copy()
                fields_dict['Membrane'] = membrane_field_array.copy()
                fields_dict['info']['pain'] = affect_state.pain
                fields_dict['info']['arousal'] = affect_state.arousal
                fields_dict['info']['valence'] = affect_state.valence
                fields_dict['info']['control'] = affect_state.control
                
                # Add brain membrane info if enabled
                if agent.cfg.brain_membrane_enabled:
                    fields_dict['info']['learning_gate'] = learning_gate
                
            field_frames.append(fields_dict)

        if done:
            break

    # Collect valence snapshot if available
    valence_snapshot = {}
    if hasattr(agent, 'valence') and isinstance(agent.valence, dict):
        valence_snapshot = dict(agent.valence)
    
    # Compute safety metrics
    bumps_per_100 = (bumps_total / max(1, steps)) * 100 if steps > 0 else 0.0
    mean_pain = np.mean(pain_history) if pain_history else 0.0
    max_pain = np.max(pain_history) if pain_history else 0.0
    mean_wall_dist = np.mean(wall_distances) if wall_distances else 0.0
    
    metrics = EpisodeMetrics(
        total_return=float(ep_ret),
        steps=steps,
        targets_collected=targets_collected,
        efficiency=float(ep_ret)/max(1, steps),
        mean_cosine=(float(np.mean(cosines)) if cosines else None),
        valence_snapshot=valence_snapshot,
        # Safety metrics
        bumps_per_100=bumps_per_100,
        mean_pain=mean_pain,
        max_pain=max_pain,
        mean_wall_distance=mean_wall_dist,
        affect_history=affect_history
    )

    episode_data = None
    if record_fields:
        episode_data = {
            'frames': field_frames,
            'world_frames': world_frames,
            'metrics': metrics
        }

    return float(ep_ret), frames, metrics, episode_data


def run_experiment(
    env_cfg: EnvConfig,
    agent_cfg: AgentConfig,
    schema_cfg: Optional[SchemaConfig],
    ablate: Ablations,
    episodes: int = 10,
    seeds: int = 1,
    base_seed: int = 0,
    use_controller: bool = False
) -> ExperimentResults:
    """
    Run multiple episodes across different seeds.
    
    Args:
        env_cfg: Environment configuration
        agent_cfg: Agent configuration
        schema_cfg: Optional schema configuration
        ablate: Ablation flags
        episodes: Number of episodes per seed
        seeds: Number of different seeds
        base_seed: Base random seed
        
    Returns:
        Experiment results
    """
    all_metrics = []
    
    for s in range(seeds):
        seed = base_seed + s
        set_global_seed(seed)
        
        # Update configs with seed
        env_cfg.seed = seed
        agent_cfg.seed = seed
        if schema_cfg:
            schema_cfg.seed = seed
        
        # Create environment and agent
        env = ForageWorld(env_cfg)
        if use_controller:
            from ..agents import FieldController, ForageAdapter
            adapter = ForageAdapter(env)
            agent = FieldController(env, adapter, agent_cfg, ablate, seed=seed)
        else:
            agent = ChemotaxisAgentCA(env, agent_cfg, ablate)
        
        # Create schema field if configured
        schema = None
        if schema_cfg and ablate.schema:
            feature_dim = 6  # [GA, GB, Novel, Vtrail, Hc, |∇P|]
            schema = SchemaField(env.H, env.W, feature_dim, schema_cfg)
        
        # Run episodes
        for ep in range(episodes):
            _, _, metrics, _ = run_episode(env, agent, schema, ablate, record_fields=False)
            metrics.seed = seed
            metrics.episode = ep
            all_metrics.append(metrics)
    
    # Compute statistics
    returns = [m.total_return for m in all_metrics]
    steps = [m.steps for m in all_metrics]
    
    results = ExperimentResults(
        metrics=all_metrics,
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        mean_steps=float(np.mean(steps)),
        std_steps=float(np.std(steps)),
        config={
            "env": asdict(env_cfg),
            "agent": asdict(agent_cfg),
            "schema": asdict(schema_cfg) if schema_cfg else None,
            "ablations": asdict(ablate)
        }
    )
    
    return results