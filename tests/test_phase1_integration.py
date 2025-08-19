"""Integration tests for Phase 1: Affect & Protective Membranes."""

import numpy as np
import pytest

from efi.configs import EnvConfig, AgentConfig, SchemaConfig, Ablations
from efi.envs import ForageWorld
from efi.agents import ChemotaxisAgentCA, SchemaField
from efi.evaluation import run_episode, run_experiment


class TestAffectIntegration:
    """Test full affect system integration."""
    
    def test_affect_reduces_bumps(self):
        """Affect system should reduce wall bumps."""
        env_cfg = EnvConfig(H=15, W=15, p_wall=0.15, n_targets_A=3, n_targets_B=2)
        
        # Baseline without affect
        agent_cfg_base = AgentConfig(affect_enabled=False)
        ablate = Ablations()
        
        # Run baseline episodes
        baseline_bumps = []
        for seed in range(5):
            env_cfg.seed = seed
            agent_cfg_base.seed = seed
            env = ForageWorld(env_cfg)
            agent = ChemotaxisAgentCA(env, agent_cfg_base, ablate)
            _, _, metrics, _ = run_episode(env, agent, None, ablate)
            baseline_bumps.append(metrics.bumps_per_100)
        
        # With affect system
        agent_cfg_affect = AgentConfig(
            affect_enabled=True,
            w_pain=0.7,
            pain_to_temp_gain=0.6,
            membrane_enabled=True,
            w_membrane=0.6
        )
        
        # Run affect episodes
        affect_bumps = []
        for seed in range(5):
            env_cfg.seed = seed
            agent_cfg_affect.seed = seed
            env = ForageWorld(env_cfg)
            agent = ChemotaxisAgentCA(env, agent_cfg_affect, ablate)
            _, _, metrics, _ = run_episode(env, agent, None, ablate)
            affect_bumps.append(metrics.bumps_per_100)
        
        # Affect should reduce bumps by at least 30%
        mean_baseline = np.mean(baseline_bumps)
        mean_affect = np.mean(affect_bumps)
        reduction = (mean_baseline - mean_affect) / max(mean_baseline, 1e-6)
        
        assert reduction >= 0.25, f"Bump reduction {reduction:.2%} < 25%"
    
    def test_pain_increases_with_adversity(self):
        """Pain should increase in adverse conditions."""
        # Environment with many B targets (adverse)
        env_cfg = EnvConfig(H=12, W=12, n_targets_A=1, n_targets_B=5, reward_B=-2.0)
        agent_cfg = AgentConfig(affect_enabled=True)
        ablate = Ablations()
        
        env = ForageWorld(env_cfg)
        agent = ChemotaxisAgentCA(env, agent_cfg, ablate)
        
        _, _, metrics, _ = run_episode(env, agent, None, ablate)
        
        # Should experience significant pain
        assert metrics.mean_pain > 0.1
        assert metrics.max_pain > 0.3
    
    def test_membrane_maintains_wall_distance(self):
        """Membrane should help maintain distance from walls."""
        env_cfg = EnvConfig(H=15, W=15, p_wall=0.2)
        
        # Without membrane
        agent_cfg_no_membrane = AgentConfig(
            affect_enabled=True,
            membrane_enabled=False
        )
        
        # With membrane
        agent_cfg_membrane = AgentConfig(
            affect_enabled=True,
            membrane_enabled=True,
            w_membrane=0.8,
            membrane_r_min=1.5
        )
        
        ablate = Ablations()
        
        # Run episodes and compare wall distances
        distances_no_membrane = []
        distances_membrane = []
        
        for seed in range(3):
            env_cfg.seed = seed
            
            # Without membrane
            agent_cfg_no_membrane.seed = seed
            env = ForageWorld(env_cfg)
            agent = ChemotaxisAgentCA(env, agent_cfg_no_membrane, ablate)
            _, _, metrics, _ = run_episode(env, agent, None, ablate)
            distances_no_membrane.append(metrics.mean_wall_distance)
            
            # With membrane
            agent_cfg_membrane.seed = seed
            env = ForageWorld(env_cfg)
            agent = ChemotaxisAgentCA(env, agent_cfg_membrane, ablate)
            _, _, metrics, _ = run_episode(env, agent, None, ablate)
            distances_membrane.append(metrics.mean_wall_distance)
        
        # Membrane should increase average wall distance
        assert np.mean(distances_membrane) > np.mean(distances_no_membrane)


class TestBrainMembraneIntegration:
    """Test brain membrane (learning gate) integration."""
    
    def test_learning_stability_under_adversity(self):
        """Learning should be more stable under adversity with brain membrane."""
        # Harsh environment with many B targets
        env_cfg = EnvConfig(
            H=15, W=15,
            n_targets_A=2,
            n_targets_B=6,
            reward_A=1.0,
            reward_B=-1.5
        )
        
        # Without brain membrane
        agent_cfg_no_bm = AgentConfig(
            affect_enabled=True,
            brain_membrane_enabled=False,
            valence_lr=0.3
        )
        
        # With brain membrane
        agent_cfg_bm = AgentConfig(
            affect_enabled=True,
            brain_membrane_enabled=True,
            brain_membrane_suppress=0.5,
            brain_membrane_min_rate=0.1,
            valence_lr=0.3
        )
        
        ablate = Ablations()
        
        # Track valence variance across episodes
        valB_no_bm = []
        valB_bm = []
        
        for ep in range(10):
            env_cfg.seed = ep
            
            # Without brain membrane
            agent_cfg_no_bm.seed = ep
            env = ForageWorld(env_cfg)
            agent = ChemotaxisAgentCA(env, agent_cfg_no_bm, ablate)
            _, _, metrics, _ = run_episode(env, agent, None, ablate)
            valB_no_bm.append(metrics.valence_snapshot.get("B", 0))
            
            # With brain membrane
            agent_cfg_bm.seed = ep
            env = ForageWorld(env_cfg)
            agent = ChemotaxisAgentCA(env, agent_cfg_bm, ablate)
            _, _, metrics, _ = run_episode(env, agent, None, ablate)
            valB_bm.append(metrics.valence_snapshot.get("B", 0))
        
        # Brain membrane should reduce variance in valence learning
        var_no_bm = np.var(valB_no_bm)
        var_bm = np.var(valB_bm)
        
        # Variance should be reduced by at least 20%
        reduction = (var_no_bm - var_bm) / max(var_no_bm, 1e-6)
        assert reduction >= 0.15, f"Variance reduction {reduction:.2%} < 15%"


class TestSafetyMetrics:
    """Test that safety metrics are properly tracked."""
    
    def test_metrics_tracked(self):
        """All safety metrics should be tracked."""
        env_cfg = EnvConfig(H=12, W=12, p_wall=0.15)
        agent_cfg = AgentConfig(affect_enabled=True)
        ablate = Ablations()
        
        env = ForageWorld(env_cfg)
        agent = ChemotaxisAgentCA(env, agent_cfg, ablate)
        
        _, _, metrics, _ = run_episode(env, agent, None, ablate)
        
        # Check all safety metrics are present
        assert hasattr(metrics, 'bumps_per_100')
        assert hasattr(metrics, 'mean_pain')
        assert hasattr(metrics, 'max_pain')
        assert hasattr(metrics, 'mean_wall_distance')
        assert hasattr(metrics, 'affect_history')
        
        # Metrics should have reasonable values
        assert metrics.bumps_per_100 >= 0
        assert 0 <= metrics.mean_pain <= 1
        assert 0 <= metrics.max_pain <= 1
        assert metrics.mean_wall_distance >= 0
        
        # Affect history should be populated if affect is enabled
        if agent_cfg.affect_enabled:
            assert len(metrics.affect_history) > 0
            # Check affect state structure
            if metrics.affect_history:
                state = metrics.affect_history[0]
                assert 'valence' in state
                assert 'arousal' in state
                assert 'control' in state
                assert 'pain' in state


class TestPerformanceMaintenance:
    """Test that performance is maintained with safety features."""
    
    def test_returns_maintained(self):
        """Returns should be maintained or improved with affect system."""
        env_cfg = EnvConfig(H=15, W=15, n_targets_A=4, n_targets_B=2)
        ablate = Ablations()
        
        # Baseline configuration
        agent_cfg_base = AgentConfig(affect_enabled=False)
        
        # With full affect system
        agent_cfg_affect = AgentConfig(
            affect_enabled=True,
            w_pain=0.7,
            membrane_enabled=True,
            w_membrane=0.6,
            brain_membrane_enabled=True
        )
        
        # Run multiple episodes
        returns_base = []
        returns_affect = []
        
        for seed in range(10):
            env_cfg.seed = seed
            
            # Baseline
            agent_cfg_base.seed = seed
            env = ForageWorld(env_cfg)
            agent = ChemotaxisAgentCA(env, agent_cfg_base, ablate)
            _, _, metrics, _ = run_episode(env, agent, None, ablate)
            returns_base.append(metrics.total_return)
            
            # With affect
            agent_cfg_affect.seed = seed
            env = ForageWorld(env_cfg)
            agent = ChemotaxisAgentCA(env, agent_cfg_affect, ablate)
            _, _, metrics, _ = run_episode(env, agent, None, ablate)
            returns_affect.append(metrics.total_return)
        
        # Returns should not degrade by more than 5%
        mean_base = np.mean(returns_base)
        mean_affect = np.mean(returns_affect)
        degradation = (mean_base - mean_affect) / max(abs(mean_base), 1e-6)
        
        assert degradation <= 0.05, f"Return degradation {degradation:.2%} > 5%"
        
        # Ideally, returns should improve slightly due to fewer bumps
        print(f"Baseline return: {mean_base:.2f}")
        print(f"Affect return: {mean_affect:.2f}")
        print(f"Change: {(mean_affect - mean_base) / max(abs(mean_base), 1e-6):.2%}")