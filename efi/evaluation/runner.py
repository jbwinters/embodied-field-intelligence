"""Episode and experiment runners."""

from typing import List, Optional, Tuple
from dataclasses import asdict

import numpy as np

from ..configs import Ablations, EnvConfig, AgentConfig, SchemaConfig
from ..envs import ForageWorld
from ..agents import ChemotaxisAgentCA, SchemaField, build_features_for_schema
from ..core import (
    corner_hazard,
    effective_potential, 
    pick_action_from_potential,
    set_global_seed,
    compose_potential
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

        # Blend frontier with decay based on local trail strength
        # High trail at current position means we've been here too much
        U = fields.get("Frontier", np.zeros_like(GA))
        if ablate.novelty:
            # Reduce frontier influence when trail is high (we're oscillating)
            trail_here = Vtrail[env.y, env.x]
            frontier_weight = max(0.0, 0.25 * (1.0 - trail_here / 3.0))
            Novel = Novel + frontier_weight * U
        
        # --- Base potential (no schema bias yet) with learned valences ---
        if hasattr(agent, 'valence') and isinstance(agent.valence, dict):
            # Channel-agnostic composition
            attractors = {"A": GA, "B": GB, "Novel": Novel}
            repulsors = {"Trail": Vtrail, "Corner": Hc}
            w_attr = {
                "A": agent.valence.get("A", 1.0),
                "B": agent.valence.get("B", 1.0),
                "Novel": 0.7
            }
            w_rep = {"Trail": 0.6, "Corner": 0.5}
            P_base = compose_potential(attractors, repulsors, w_attr, w_rep, bias=None)
        else:
            # Legacy method for backwards compatibility
            P_base = effective_potential(GA, GB, Novel, Vtrail, Hc,
                                        wA=agent.valA,
                                        wB=agent.valB,
                                        wN=0.7,
                                        kV=0.6, kH=0.5)

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

        # Use trail strength as natural "stuck" signal
        # High trail means we've been here too much
        trail_here = Vtrail[env.y, env.x]
        if trail_here > 2.0:  # We're oscillating/stuck
            # Temperature based on how stuck we are
            temp = 0.5 + (trail_here - 2.0) * 0.5
            temp = min(temp, 2.0)  # Cap at 2.0
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

        # Step env
        obs, r, done, info = env.step(a)
        
        # Compute gradient-motion alignment (after step)
        dy, dx = env.y - prev_yx[0], env.x - prev_yx[1]
        if dy != 0 or dx != 0:  # Only if we moved
            gy, gx = np.gradient(P_eff.astype(np.float32))
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
        if not info.get("moved", False):
            agent.stuck_count += 1
        else:
            agent.stuck_count = 0
        agent.last_pos = (env.y, env.x)

        # Learn from pickups
        picked = info.get("picked")
        if picked == "A":
            targets_collected["A"] += 1
            agent.learn_valence("A", env.cfg.reward_A)
        elif picked == "B":
            targets_collected["B"] += 1
            agent.learn_valence("B", env.cfg.reward_B)

        # Optionally record frames & fields
        if record:
            world_rgb = env.render_rgb()
            frames.append(world_rgb.copy())

        if record_fields:
            world_rgb = env.render_rgb()
            world_frames.append(world_rgb.copy())
            field_frames.append({
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
            })

        if done:
            break

    # Collect valence snapshot if available
    valence_snapshot = {}
    if hasattr(agent, 'valence') and isinstance(agent.valence, dict):
        valence_snapshot = dict(agent.valence)
    
    metrics = EpisodeMetrics(
        total_return=float(ep_ret),
        steps=steps,
        targets_collected=targets_collected,
        efficiency=float(ep_ret)/max(1, steps),
        mean_cosine=(float(np.mean(cosines)) if cosines else None),
        valence_snapshot=valence_snapshot
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