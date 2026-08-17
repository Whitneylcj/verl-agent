import numpy as np

from recipe.exact.env_probes import (
    alfworld_factor_snapshot,
    appworld_factor_snapshot,
    conservative_future_schema,
    sokoban_factor_snapshot,
    webshop_factor_snapshot,
)


def test_sokoban_target_bits_have_stable_coordinate_ids():
    fixed = np.array([[0, 0, 0], [0, 2, 2], [0, 0, 0]])
    state = np.array([[0, 0, 0], [0, 3, 4], [0, 0, 0]])
    snapshot = sokoban_factor_snapshot(fixed, state)
    assert snapshot["factor_ids"] == ("target:1:1", "target:1:2")
    assert snapshot["values"] == (1.0, 0.0)


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
        "object_acquired",
        "object_cleaned",
        "object_placed_1",
        "goal_satisfied",
    )
    assert snapshot["values"] == (1.0, 1.0, 1.0, 0.0)


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
    assert held["values"][0] == 1.0
    assert released["values"][0] == 0.0


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
