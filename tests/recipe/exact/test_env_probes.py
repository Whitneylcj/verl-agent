import numpy as np
import pytest

from recipe.exact.env_probes import (
    alfworld_factor_snapshot,
    appworld_factor_snapshot,
    conservative_future_schema,
    sokoban_factor_snapshot,
    webshop_factor_snapshot,
)

SOKOBAN_REWARD_WEIGHTS = {
    "step_penalty": -0.1,
    "box_on_target_reward": 1.0,
    "box_off_target_penalty": -1.0,
    "all_boxes_on_target_reward": 10.0,
}


def test_sokoban_snapshot_uses_only_official_reward_events():
    event_counts = {
        "step_penalty": 4,
        "box_on_target_reward": 2,
        "box_off_target_penalty": 1,
        "all_boxes_on_target_reward": 1,
    }
    snapshot = sokoban_factor_snapshot(event_counts, SOKOBAN_REWARD_WEIGHTS)
    assert snapshot["factor_ids"] == (
        "step_penalty",
        "box_on_target_reward",
        "box_off_target_penalty",
        "all_boxes_on_target_reward",
    )
    assert snapshot["values"] == (4.0, 2.0, 1.0, 1.0)
    assert snapshot["potential_weights"] == (-0.1, 1.0, -1.0, 10.0)
    assert snapshot["schema_version"] == "exact.sokoban.official_reward.v1"


def test_sokoban_official_factor_deltas_reconstruct_each_step_reward():
    checkpoints = (
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (2, 1, 0, 0),
        (3, 1, 1, 0),
        (4, 2, 1, 1),
    )
    weights = np.asarray(tuple(SOKOBAN_REWARD_WEIGHTS.values()))
    values = np.asarray(checkpoints, dtype=np.float64)
    reconstructed = np.diff(values, axis=0) @ weights
    np.testing.assert_allclose(reconstructed, (-0.1, 0.9, -1.1, 10.9))


def test_sokoban_probe_rejects_non_official_or_invalid_factors():
    with pytest.raises(ValueError, match="missing=.*all_boxes_on_target_reward"):
        sokoban_factor_snapshot(
            {"step_penalty": 0, "box_on_target_reward": 0, "box_off_target_penalty": 0},
            SOKOBAN_REWARD_WEIGHTS,
        )
    with pytest.raises(ValueError, match="non-negative integers"):
        sokoban_factor_snapshot(
            {
                "step_penalty": 0.5,
                "box_on_target_reward": 0,
                "box_off_target_penalty": 0,
                "all_boxes_on_target_reward": 0,
            },
            SOKOBAN_REWARD_WEIGHTS,
        )


def test_alfworld_goal_probe_is_available_from_sparse_info():
    assert alfworld_factor_snapshot({"won": True})["values"] == (1.0,)
    assert alfworld_factor_snapshot({})["values"] == (0.0,)


def test_alfworld_task_probe_compiles_facts_into_subgoals():
    task = {
        "task_type": "pick_clean_then_place_in_recep",
        "object_target": "Apple",
        "parent_target": "CounterTop",
        "toggle_target": "",
    }
    info = {
        "won": False,
        "facts": [
            {"name": "holds", "arguments": ["agent", "apple 1"]},
            {"name": "isclean", "arguments": ["apple 1"]},
            {"name": "inreceptacle", "arguments": ["apple 1", "countertop 2"]},
        ],
    }
    snapshot = alfworld_factor_snapshot(info, task)
    assert snapshot["factor_ids"] == (
        "object_discovered",
        "inventory_target_compatible",
        "object_acquired",
        "object_cleaned",
        "object_placed_1",
        "goal_satisfied",
    )
    assert snapshot["values"] == (0.0, 1.0, 1.0, 1.0, 1.0, 0.0)


def test_alfworld_probe_tracks_observable_discovery_without_task_text_leakage():
    task = {
        "task_type": "pick_heat_then_place_in_recep",
        "object_target": "Mug",
        "parent_target": "CoffeeMachine",
    }
    reset = alfworld_factor_snapshot(
        {"observation_text": "You see a fridge 1. Your task is to: heat some mug."},
        task,
    )
    visible = alfworld_factor_snapshot(
        {"observation_text": "The fridge is open. In it, you see a mug 1."},
        task,
    )
    remembered = alfworld_factor_snapshot(
        {"observation_text": "You arrive at coffeemachine 1.", "exact.object_discovered": True},
        task,
    )
    assert reset["values"][0] == 0.0
    assert visible["values"][0] == 1.0
    assert remembered["values"][0] == 1.0


def test_alfworld_probe_penalizes_only_wrong_inventory_objects():
    task = {
        "task_type": "pick_heat_then_place_in_recep",
        "object_target": "Mug",
        "parent_target": "CoffeeMachine",
    }
    wrong = alfworld_factor_snapshot(
        {"facts": [{"name": "holds", "arguments": ["agent", "bowl 1"]}]},
        task,
    )
    target = alfworld_factor_snapshot(
        {"facts": [{"name": "holds", "arguments": ["agent", "mug 1"]}]},
        task,
    )
    assert wrong["values"][1] == 0.0
    assert target["values"][1] == 1.0


def test_alfworld_pick_two_counts_distinct_object_instances():
    task = {
        "task_type": "pick_two_obj_and_place",
        "object_target": "Apple",
        "parent_target": "CounterTop",
    }
    info = {
        "facts": [
            {"name": "inreceptacle", "arguments": ["apple 1", "countertop 1"]},
            {"name": "inreceptacle", "arguments": ["apple 2", "countertop 1"]},
        ]
    }
    snapshot = alfworld_factor_snapshot(info, task)
    assert snapshot["values"][-3:-1] == (1.0, 1.0)


def test_alfworld_probe_reports_current_state_regressions():
    task = {
        "task_type": "pick_and_place_simple",
        "object_target": "Apple",
        "parent_target": "CounterTop",
    }
    held = alfworld_factor_snapshot(
        {"facts": [{"name": "holds", "arguments": ["agent", "apple 1"]}]},
        task,
    )
    released = alfworld_factor_snapshot({"facts": []}, task)
    acquired_index = held["factor_ids"].index("object_acquired")
    assert held["values"][acquired_index] == 1.0
    assert released["values"][acquired_index] == 0.0


def test_webshop_missing_components_are_zero_not_unknown_schema():
    initial = webshop_factor_snapshot(None)
    terminal = webshop_factor_snapshot({"r_type": 1, "r_att": 0.5, "r_price": True})
    assert initial["factor_ids"] == terminal["factor_ids"]
    assert terminal["values"] == (1.0, 0.5, 0.0, 1.0)


def test_appworld_per_test_boolean_schema_is_fixed():
    evaluation = {
        "success": False,
        "passes": [
            {"requirement": "created the playlist"},
        ],
        "failures": [
            {"requirement": "shared the playlist", "traceback": "changes over time"},
        ],
    }
    first = appworld_factor_snapshot(evaluation)
    second = appworld_factor_snapshot(evaluation, expected_factor_ids=first["factor_ids"])
    assert first == second
    progressed = {
        "success": True,
        "passes": [
            {"requirement": "created the playlist"},
            {"requirement": "shared the playlist"},
        ],
        "failures": [],
    }
    progressed_snapshot = appworld_factor_snapshot(
        progressed,
        expected_factor_ids=first["factor_ids"],
    )
    assert progressed_snapshot["values"] == (1.0, 1.0)

    changed = {
        "success": False,
        "passes": [{"requirement": "created the playlist"}],
        "failures": [],
    }
    try:
        appworld_factor_snapshot(changed, expected_factor_ids=first["factor_ids"])
    except ValueError as error:
        assert "schema changed" in str(error)
    else:
        raise AssertionError("AppWorld probe accepted a changed verifier schema")


def test_default_effect_schema_keeps_every_future_factor():
    snapshot = webshop_factor_snapshot(None)
    schema = conservative_future_schema(snapshot, "webshop")
    assert schema["descendant_factor_ids"] == snapshot["factor_ids"]
    assert schema["opaque"] is False
