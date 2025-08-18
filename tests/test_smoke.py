#!/usr/bin/env python3
"""Smoke tests to ensure both agent paths work correctly."""

from efi.configs import EnvConfig, AgentConfig, SchemaConfig, Ablations
from efi.evaluation import run_experiment


def test_chemotaxis_agent():
    """Test baseline chemotaxis agent (no controller)."""
    print("Testing ChemotaxisAgentCA...")
    
    res = run_experiment(
        env_cfg=EnvConfig(max_steps=10, seed=42),
        agent_cfg=AgentConfig(seed=42),
        schema_cfg=SchemaConfig(seed=42),
        ablate=Ablations(),
        episodes=2,
        seeds=1,
        use_controller=False,
    )
    
    assert len(res.metrics) == 2, f"Expected 2 metrics, got {len(res.metrics)}"
    
    # Ensure valence snapshot exists and is a dict
    for m in res.metrics:
        assert isinstance(m.valence_snapshot, dict), "Valence snapshot should be a dict"
        assert "A" in m.valence_snapshot, "Should have A valence"
        assert "B" in m.valence_snapshot, "Should have B valence"
        assert "Novel" in m.valence_snapshot, "Should have Novel valence"
    
    # Check that valences can change if pickups happened
    if res.metrics[0].targets_collected.get("A", 0) > 0:
        # A valence should have increased
        initial_valA = AgentConfig().valA_init
        final_valA = res.metrics[0].valence_snapshot["A"]
        assert final_valA != initial_valA, "A valence should change after pickup"
    
    print(f"  ✓ ChemotaxisAgentCA: {res.metrics[0].total_return:.2f} return, "
          f"{res.metrics[0].steps} steps")
    print(f"  ✓ Valences: {res.metrics[0].valence_snapshot}")


def test_field_controller():
    """Test FieldController path."""
    print("\nTesting FieldController...")
    
    res = run_experiment(
        env_cfg=EnvConfig(max_steps=10, seed=43),
        agent_cfg=AgentConfig(seed=43),
        schema_cfg=SchemaConfig(seed=43),
        ablate=Ablations(),
        episodes=2,
        seeds=1,
        use_controller=True,
    )
    
    assert len(res.metrics) == 2, f"Expected 2 metrics, got {len(res.metrics)}"
    
    # Ensure valence snapshot exists
    for m in res.metrics:
        assert isinstance(m.valence_snapshot, dict), "Valence snapshot should be a dict"
        assert "A" in m.valence_snapshot, "Should have A valence"
        assert "B" in m.valence_snapshot, "Should have B valence"
        assert "Novel" in m.valence_snapshot, "Should have Novel valence"
    
    print(f"  ✓ FieldController: {res.metrics[0].total_return:.2f} return, "
          f"{res.metrics[0].steps} steps")
    print(f"  ✓ Valences: {res.metrics[0].valence_snapshot}")


def test_valence_learning():
    """Test that valence learning works with negative B rewards."""
    print("\nTesting valence learning with B as undesirable...")
    
    # Configure B as undesirable
    env_cfg = EnvConfig(
        max_steps=50,
        n_targets_A=2,
        n_targets_B=4,
        reward_A=1.0,
        reward_B=-1.0,
        seed=44
    )
    
    agent_cfg = AgentConfig(
        valA_init=1.0,
        valB_init=0.2,  # Starts positive
        valence_lr=0.3,
        seed=44
    )
    
    # Test both agent types
    for use_controller, name in [(False, "ChemotaxisAgentCA"), (True, "FieldController")]:
        print(f"  Testing {name}...")
        
        res = run_experiment(
            env_cfg=env_cfg,
            agent_cfg=agent_cfg,
            schema_cfg=None,  # No schema for this test
            ablate=Ablations(schema=0),
            episodes=1,
            seeds=1,
            use_controller=use_controller,
        )
        
        m = res.metrics[0]
        initial_valB = agent_cfg.valB_init
        final_valB = m.valence_snapshot["B"]
        
        # If any B was collected, valence should decrease
        if m.targets_collected.get("B", 0) > 0:
            assert final_valB < initial_valB, \
                f"{name}: B valence should decrease after negative reward"
            print(f"    ✓ B avoidance learned: {initial_valB:.2f} → {final_valB:.2f}")
        else:
            print(f"    ✓ No B collected, valence unchanged: {final_valB:.2f}")


def test_compose_p_delegation():
    """Test that runner delegates to agent.compose_P when available."""
    print("\nTesting compose_P delegation...")
    
    # FieldController has compose_P
    res = run_experiment(
        env_cfg=EnvConfig(max_steps=5, seed=45),
        agent_cfg=AgentConfig(seed=45),
        schema_cfg=SchemaConfig(seed=45),
        ablate=Ablations(),
        episodes=1,
        seeds=1,
        use_controller=True,
    )
    
    # Should complete without errors
    assert len(res.metrics) == 1
    print(f"  ✓ FieldController.compose_P delegation works")
    
    # ChemotaxisAgentCA doesn't have compose_P, uses legacy path
    res = run_experiment(
        env_cfg=EnvConfig(max_steps=5, seed=46),
        agent_cfg=AgentConfig(seed=46),
        schema_cfg=SchemaConfig(seed=46),
        ablate=Ablations(),
        episodes=1,
        seeds=1,
        use_controller=False,
    )
    
    assert len(res.metrics) == 1
    print(f"  ✓ Legacy composition path works")


def test_config_weights():
    """Test that weights are read from config."""
    print("\nTesting config weight usage...")
    
    # Create config with custom weights
    agent_cfg = AgentConfig(
        w_novel=0.9,  # Different from default
        w_trail=0.8,
        w_corner=0.3,
        seed=47
    )
    
    env_cfg = EnvConfig(max_steps=5, seed=47)
    
    # Test FieldController uses config weights
    from efi.envs import ForageWorld
    from efi.agents import FieldController, ForageAdapter
    
    env = ForageWorld(env_cfg)
    adapter = ForageAdapter(env)
    agent = FieldController(env, adapter, agent_cfg, Ablations())
    
    # Check that valence includes the config weight
    assert agent.valence["Novel"] == 0.9, f"Novel weight should be {agent_cfg.w_novel}"
    
    # Check that compose_P uses config weights (indirectly via cfg reference)
    assert agent.cfg.w_trail == 0.8
    assert agent.cfg.w_corner == 0.3
    
    print(f"  ✓ Config weights loaded: Novel={agent.valence['Novel']}, "
          f"Trail={agent.cfg.w_trail}, Corner={agent.cfg.w_corner}")


def main():
    """Run all smoke tests."""
    print("="*60)
    print("Running smoke tests for EFI")
    print("="*60)
    
    test_chemotaxis_agent()
    test_field_controller()
    test_valence_learning()
    test_compose_p_delegation()
    test_config_weights()
    
    print("\n" + "="*60)
    print("✅ All smoke tests passed!")
    print("="*60)


if __name__ == "__main__":
    main()