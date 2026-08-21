import json

import pytest

from recipe.exact.alfworld_verifier import (
    ALFWORLD_TASK_FACTOR_IDS,
    AlfworldVerifierSpec,
    evaluate_alfworld_goal_factors,
    load_alfworld_verifier_spec,
)


class _Variable:
    def __init__(self, name):
        self.name = name


class _Fact:
    def __init__(self, name, *arguments):
        self.name = name
        self.arguments = tuple(_Variable(argument) for argument in arguments)


def _placement_facts(*, count=0, state_predicate=None):
    facts = [
        _Fact("objectType", "apple_1", "AppleType"),
        _Fact("objectType", "apple_2", "apple-type"),
        _Fact("receptacleType", "bowl_1", "BowlType"),
    ]
    for index in range(count):
        facts.append(_Fact("inReceptacle", f"apple_{index + 1}", "bowl_1"))
    if state_predicate is not None:
        capability, state = state_predicate
        facts.extend((_Fact(capability, "apple_1"), _Fact(state, "apple_1")))
    return facts


@pytest.mark.parametrize(
    ("task_type", "facts", "won", "expected"),
    (
        (
            "pick_and_place_simple",
            _placement_facts(count=1),
            True,
            {"target_placed": 1.0},
        ),
        (
            "pick_two_obj_and_place",
            _placement_facts(count=1),
            False,
            {"one_target_placed": 1.0, "two_targets_placed": 0.0},
        ),
        (
            "pick_two_obj_and_place",
            _placement_facts(count=2),
            True,
            {"one_target_placed": 1.0, "two_targets_placed": 1.0},
        ),
    ),
)
def test_placement_goal_factors_match_official_pddl(task_type, facts, won, expected):
    spec = AlfworldVerifierSpec(task_type, "Apple", receptacle_type="Bowl")
    assert evaluate_alfworld_goal_factors(spec, facts, won) == expected


@pytest.mark.parametrize(
    ("task_type", "capability", "state", "state_factor", "joint_factor"),
    (
        (
            "pick_clean_then_place_in_recep",
            "cleanable",
            "isClean",
            "target_clean",
            "target_clean_and_placed",
        ),
        (
            "pick_heat_then_place_in_recep",
            "heatable",
            "isHot",
            "target_hot",
            "target_hot_and_placed",
        ),
        (
            "pick_cool_then_place_in_recep",
            "coolable",
            "isCool",
            "target_cool",
            "target_cool_and_placed",
        ),
    ),
)
def test_state_change_goal_factors_preserve_same_object_joint_condition(
    task_type,
    capability,
    state,
    state_factor,
    joint_factor,
):
    spec = AlfworldVerifierSpec(task_type, "Apple", receptacle_type="Bowl")
    state_only = _placement_facts(count=0, state_predicate=(capability, state))
    assert evaluate_alfworld_goal_factors(spec, state_only, False) == {
        state_factor: 1.0,
        "target_placed": 0.0,
        joint_factor: 0.0,
    }
    state_and_placed = _placement_facts(
        count=1,
        state_predicate=(capability, state),
    )
    assert evaluate_alfworld_goal_factors(spec, state_and_placed, True) == {
        state_factor: 1.0,
        "target_placed": 1.0,
        joint_factor: 1.0,
    }


def test_look_goal_matches_held_object_and_toggled_light_at_agent_location():
    spec = AlfworldVerifierSpec(
        "look_at_obj_in_light",
        "Apple",
        toggle_type="DeskLamp",
    )
    facts = [
        _Fact("objectType", "apple_1", "AppleType"),
        _Fact("objectType", "lamp_1", "DeskLampType"),
        _Fact("toggleable", "lamp_1"),
        _Fact("isToggled", "lamp_1"),
        _Fact("inReceptacle", "lamp_1", "desk_1"),
        _Fact("receptacleAtLocation", "desk_1", "room_1"),
        _Fact("atLocation", "agent_1", "room_1"),
    ]
    assert evaluate_alfworld_goal_factors(spec, facts, False) == {
        "target_held": 0.0,
        "light_ready_at_agent": 1.0,
    }
    facts.append(_Fact("holds", "agent_1", "apple_1"))
    assert evaluate_alfworld_goal_factors(spec, facts, True) == {
        "target_held": 1.0,
        "light_ready_at_agent": 1.0,
    }


def test_goal_factor_evaluator_fails_closed_on_official_won_mismatch():
    spec = AlfworldVerifierSpec(
        "pick_and_place_simple",
        "Apple",
        receptacle_type="Bowl",
    )
    with pytest.raises(RuntimeError, match="disagree with the official PDDL won"):
        evaluate_alfworld_goal_factors(spec, _placement_facts(count=1), False)


def test_verifier_spec_loads_only_supported_unsliced_text_tasks(tmp_path):
    gamefile = tmp_path / "game.tw-pddl"
    gamefile.write_text("", encoding="utf-8")
    metadata_path = tmp_path / "traj_data.json"
    metadata_path.write_text(
        json.dumps(
            {
                "task_type": "look_at_obj_in_light",
                "pddl_params": {
                    "object_target": "Apple",
                    "toggle_target": "DeskLamp",
                    "object_sliced": False,
                },
            }
        ),
        encoding="utf-8",
    )
    spec = load_alfworld_verifier_spec(gamefile)
    assert spec == AlfworldVerifierSpec(
        "look_at_obj_in_light",
        "Apple",
        toggle_type="DeskLamp",
    )
    assert spec.factor_ids == ALFWORLD_TASK_FACTOR_IDS["look_at_obj_in_light"]

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["pddl_params"]["object_sliced"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="sliced-object"):
        load_alfworld_verifier_spec(gamefile)
