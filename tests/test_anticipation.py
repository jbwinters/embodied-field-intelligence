"""Behavioral and causal tests of learned temporal control."""

import ast
from pathlib import Path

import numpy as np
import pytest

from efi.agents.motion_schema import MotionSchema
from efi.core.anticipation import arrival_values, action_probabilities
from efi.envs.crossing_world import CrossingConfig, CrossingWorld
from efi.evaluation.crossing import make_crossing_agent, crossing_experiment


def sighting(model, point, walls=None, visible=None, learn=True):
    walls = np.zeros(model.shape, dtype=bool) if walls is None else walls
    visible = np.ones(model.shape, dtype=bool) if visible is None else visible
    occupied = np.zeros(model.shape, dtype=bool)
    occupied[point] = True
    model.observe(occupied, visible, walls, learn=learn)


def teach_straight_motion(model, repeats=10):
    for _ in range(repeats):
        model.reset()
        for x in range(2, 8):
            sighting(model, (5, x))


def test_motion_is_learned_and_prediction_is_scored_before_update():
    model = MotionSchema((11, 13))
    teach_straight_motion(model)
    model.reset()
    sighting(model, (5, 3))
    sighting(model, (5, 4))
    forecast = model.forecast(np.zeros(model.shape, bool), 3)
    assert forecast[0][5, 5] > 0.95
    assert forecast[2][5, 7] > 0.85
    sighting(model, (5, 3))  # reversal contradicts the learned continuation
    assert model.last_loss > 3
    # New experience changes the rule, even after a long stable history.
    for _ in range(30):
        sighting(model, (5, 4))
        sighting(model, (5, 3))
    assert model.forecast(np.zeros(model.shape, bool), 1)[0][5, 4] > 0.9
    assert model.last_loss < 0.2


def test_unobserved_truth_cannot_train_or_seed_motion():
    model = MotionSchema((11, 13))
    visible = np.zeros(model.shape, bool)
    visible[3:8, 3:8] = True
    for x in range(3):
        sighting(model, (0, x), visible=visible)
    assert not model.mass.any()
    assert model.transitions == 0


def test_ambiguous_visible_association_is_not_learned():
    model = MotionSchema((7, 7))
    sighting(model, (3, 3))
    occupied = np.zeros(model.shape, bool)
    occupied[3, 2] = occupied[3, 4] = True
    model.observe(occupied, np.ones(model.shape, bool), np.zeros(model.shape, bool))
    assert model.transitions == 0
    assert np.all(model.velocity == -1)


def test_forecast_light_cone_walls_mass_and_uncertainty():
    model = MotionSchema((15, 15))
    walls = np.zeros(model.shape, bool)
    walls[:, 9] = True
    sighting(model, (7, 7), walls)
    yy, xx = np.indices(model.shape)
    for h, field in enumerate(model.forecast(walls, 4), 1):
        assert np.all(field[np.abs(yy - 7) + np.abs(xx - 7) > h] == 0)
        assert np.all(field[:, 9:] == 0)
        assert field.sum() == pytest.approx(1, abs=1e-5)
        assert field.max() < 1  # uniform prior does not invent a direction


def test_future_arrival_hazard_changes_current_action_and_wait_is_available():
    walls = np.ones((3, 5), bool)
    walls[1, :] = False
    terminal = np.tile(np.arange(5, dtype=np.float32), (3, 1))
    costs = np.full((3, 5), 0.01, np.float32)
    safe = np.zeros_like(terminal)
    danger = safe.copy()
    danger[1, 2] = 1
    free_scores = arrival_values(terminal, costs, [safe, safe], walls, 0.02, 10)
    risky_scores = arrival_values(terminal, costs, [danger, safe], walls, 0.02, 10)
    free = action_probabilities(free_scores[:, 1, 1], 0.02)
    risky = action_probabilities(risky_scores[:, 1, 1], 0.02)
    assert free[3] > 0.99
    assert risky[3] < 0.001
    assert risky[4] > 0.99


def test_swap_collision_has_an_edge_cost():
    walls = np.ones((3, 4), bool)
    walls[1] = False
    terminal = np.tile(np.arange(4, dtype=np.float32), (3, 1))
    costs = np.zeros_like(terminal)
    hazard = np.zeros_like(terminal)
    edge = np.zeros((5, 3, 4), dtype=np.float32)
    edge[3, 1, 1] = 1
    scores = arrival_values(terminal, costs, [hazard], walls, 0.02, 10, [edge])
    assert action_probabilities(scores[:, 1, 1], 0.02)[3] < 0.001


def test_intentional_wait_preserves_pose_and_does_not_trigger_recovery():
    env = CrossingWorld(CrossingConfig(seed=4))
    agent = make_crossing_agent()
    obs = env.reset()
    agent.reset()
    agent.observe(obs)
    pose = agent.pose
    obs, _, _, info = env.step(4)
    assert not info["moved"] and not info["bump"]
    agent.after_env_step(4, False, None)
    agent.observe(obs)  # predictive schema must also accept a wait transition
    assert agent.pose == pose
    assert agent._search_boost_ticks == 0


def test_environment_collision_timing_and_reproducibility():
    a, b = CrossingWorld(CrossingConfig(seed=3)), CrossingWorld(CrossingConfig(seed=3))
    assert np.array_equal(a.reset(), b.reset())
    for action in (4, 2, 3, 4, 4):
        left, right = a.step(action), b.step(action)
        assert np.array_equal(left[0], right[0])
        assert left[1:] == right[1:]
    a.hazard_index = a.H // 2 - 2
    a.direction = 1
    a.rule = "continue"
    a.y, a.x = a.H // 2, a.W // 2 - 1
    _, reward, done, info = a.step(3)
    assert done and info["collision"] and reward < -1


def test_controller_has_no_environment_or_rule_access():
    root = Path(__file__).resolve().parents[1]
    for filename in ("anticipatory_controller.py", "motion_schema.py"):
        tree = ast.parse((root / "efi" / "agents" / filename).read_text())
        forbidden = {"env", "hazard_index", "direction", "rule", "lane"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden


def test_learning_adds_crossing_capability_on_unseen_seeds():
    result = crossing_experiment(seeds=4, episodes=12, base_seed=2000, record=False)
    for phase in ("acquire", "transfer", "reverse"):
        s = result["summary"][phase]
        assert s["learned"]["success"] >= 0.8
        assert s["learned"]["success"] > s["static"]["success"] + 0.15
        assert s["learned"]["success"] > s["unlearned"]["success"] + 0.10
        assert s["learned"]["return"] > s["static"]["return"]
    assert result["summary"]["acquire"]["frozen"] == result["summary"]["acquire"]["learned"]
    assert result["summary"]["reverse"]["learned"]["success"] > (
        result["summary"]["reverse"]["frozen"]["success"] + 0.15
    )


def test_future_costs_cannot_outrun_backward_horizon():
    walls = np.zeros((11, 11), bool)
    terminal = np.zeros(walls.shape, np.float32)
    costs = np.full(walls.shape, 0.01, np.float32)
    empty = np.zeros_like(terminal)
    hazard = empty.copy()
    hazard[5, 8] = 1
    baseline = arrival_values(terminal, costs, [empty, empty], walls, 0.02, 2)
    changed = arrival_values(terminal, costs, [empty, hazard], walls, 0.02, 2)
    assert np.array_equal(baseline[:, 5, 5], changed[:, 5, 5])
    assert not np.array_equal(baseline[:, 5, 7], changed[:, 5, 7])
