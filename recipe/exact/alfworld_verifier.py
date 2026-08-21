"""Official PDDL goal-condition probes for the ALFWorld text environment."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

SUPPORTED_ALFWORLD_TASKS = (
    "pick_and_place_simple",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
)

ALFWORLD_TASK_FACTOR_IDS = {
    "pick_and_place_simple": ("target_placed",),
    "pick_two_obj_and_place": ("one_target_placed", "two_targets_placed"),
    "look_at_obj_in_light": ("target_held", "light_ready_at_agent"),
    "pick_clean_then_place_in_recep": (
        "target_clean",
        "target_placed",
        "target_clean_and_placed",
    ),
    "pick_heat_then_place_in_recep": (
        "target_hot",
        "target_placed",
        "target_hot_and_placed",
    ),
    "pick_cool_then_place_in_recep": (
        "target_cool",
        "target_placed",
        "target_cool_and_placed",
    ),
}

_PLACEMENT_READS = (
    "alfworld.pddl.objecttype",
    "alfworld.pddl.receptacletype",
    "alfworld.pddl.inreceptacle",
)
_LOOK_READ_SETS = {
    "target_held": ("alfworld.pddl.objecttype", "alfworld.pddl.holds"),
    "light_ready_at_agent": (
        "alfworld.pddl.objecttype",
        "alfworld.pddl.toggleable",
        "alfworld.pddl.istoggled",
        "alfworld.pddl.inreceptacle",
        "alfworld.pddl.receptacleatlocation",
        "alfworld.pddl.atlocation",
    ),
}


def _state_read_sets(state_name: str) -> dict[str, tuple[str, ...]]:
    capability = {
        "clean": "alfworld.pddl.cleanable",
        "hot": "alfworld.pddl.heatable",
        "cool": "alfworld.pddl.coolable",
    }[state_name]
    state = {
        "clean": "alfworld.pddl.isclean",
        "hot": "alfworld.pddl.ishot",
        "cool": "alfworld.pddl.iscool",
    }[state_name]
    state_factor = f"target_{state_name}"
    return {
        state_factor: ("alfworld.pddl.objecttype", capability, state),
        "target_placed": _PLACEMENT_READS + (capability,),
        f"{state_factor}_and_placed": _PLACEMENT_READS + (capability, state),
    }


ALFWORLD_TASK_FACTOR_READ_SETS = {
    "pick_and_place_simple": {"target_placed": _PLACEMENT_READS},
    "pick_two_obj_and_place": {
        "one_target_placed": _PLACEMENT_READS,
        "two_targets_placed": _PLACEMENT_READS,
    },
    "look_at_obj_in_light": _LOOK_READ_SETS,
    "pick_clean_then_place_in_recep": _state_read_sets("clean"),
    "pick_heat_then_place_in_recep": _state_read_sets("hot"),
    "pick_cool_then_place_in_recep": _state_read_sets("cool"),
}


@dataclass(frozen=True)
class AlfworldVerifierSpec:
    task_type: str
    object_type: str
    receptacle_type: str | None = None
    toggle_type: str | None = None

    @property
    def factor_ids(self) -> tuple[str, ...]:
        return ALFWORLD_TASK_FACTOR_IDS[self.task_type]


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"ALFWorld verifier requires non-empty {field}")
    return value.strip()


def load_alfworld_verifier_spec(gamefile: str | Path) -> AlfworldVerifierSpec:
    """Load verifier-only task parameters from the official sibling metadata."""

    gamefile = Path(gamefile)
    if gamefile.name != "game.tw-pddl":
        raise ValueError("ALFWorld verifier requires a game.tw-pddl path")
    metadata_path = gamefile.with_name("traj_data.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing ALFWorld trajectory metadata: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, Mapping):
        raise TypeError("ALFWorld traj_data.json must contain an object")

    task_type = _nonempty_string(metadata.get("task_type"), "task_type")
    if task_type not in SUPPORTED_ALFWORLD_TASKS:
        raise ValueError(f"unsupported ALFWorld verifier task type: {task_type}")
    params = metadata.get("pddl_params")
    if not isinstance(params, Mapping):
        raise TypeError("ALFWorld verifier requires pddl_params metadata")
    if bool(params.get("object_sliced", False)):
        raise ValueError("sliced-object ALFWorld tasks are not supported by the text environment")

    object_type = _nonempty_string(params.get("object_target"), "object_target")
    receptacle_type = None
    toggle_type = None
    if task_type != "look_at_obj_in_light":
        receptacle_type = _nonempty_string(params.get("parent_target"), "parent_target")
    else:
        toggle_type = _nonempty_string(params.get("toggle_target"), "toggle_target")
    return AlfworldVerifierSpec(
        task_type=task_type,
        object_type=object_type,
        receptacle_type=receptacle_type,
        toggle_type=toggle_type,
    )


def _canonical_type(value: Any) -> str:
    canonical = re.sub(r"[^a-z0-9]", "", str(value).lower())
    return canonical.removesuffix("type")


def _fact_index(facts: Iterable[Any]) -> dict[str, set[tuple[str, ...]]]:
    index: dict[str, set[tuple[str, ...]]] = {}
    for fact in facts:
        name = getattr(fact, "name", None)
        arguments = getattr(fact, "arguments", None)
        if not isinstance(name, str) or arguments is None:
            raise TypeError("ALFWorld facts must expose name and arguments")
        values = []
        for argument in arguments:
            value = getattr(argument, "name", argument)
            if not isinstance(value, str) or not value:
                raise TypeError("ALFWorld fact arguments must have non-empty names")
            values.append(value.lower())
        index.setdefault(name.lower(), set()).add(tuple(values))
    if not index:
        raise ValueError("ALFWorld verifier received an empty PDDL fact set")
    return index


def _typed_entities(
    index: Mapping[str, set[tuple[str, ...]]],
    predicate: str,
    target_type: str,
) -> set[str]:
    canonical_target = _canonical_type(target_type)
    rows = index.get(predicate, set())
    if any(len(row) != 2 for row in rows):
        raise ValueError(f"ALFWorld {predicate} facts must have exactly two arguments")
    entities = {
        entity
        for entity, entity_type in rows
        if _canonical_type(entity_type) == canonical_target
    }
    if not entities:
        raise ValueError(f"ALFWorld verifier found no {predicate} entity for the target type")
    return entities


def _unary_entities(index: Mapping[str, set[tuple[str, ...]]], predicate: str) -> set[str]:
    return {row[0] for row in index.get(predicate, set()) if len(row) == 1}


def evaluate_alfworld_goal_factors(
    spec: AlfworldVerifierSpec,
    facts: Iterable[Any],
    won: Any,
) -> dict[str, float]:
    """Evaluate official task goal conditions without reading observation text."""

    if spec.task_type not in SUPPORTED_ALFWORLD_TASKS:
        raise ValueError(f"unsupported ALFWorld verifier task type: {spec.task_type}")
    won_value = float(won)
    if won_value not in {0.0, 1.0}:
        raise ValueError("ALFWorld won must be binary")

    index = _fact_index(facts)
    target_objects = _typed_entities(index, "objecttype", spec.object_type)
    in_receptacle = index.get("inreceptacle", set())
    factors: dict[str, float]

    if spec.task_type == "look_at_obj_in_light":
        if spec.toggle_type is None:
            raise ValueError("look_at_obj_in_light requires toggle_type")
        target_toggles = _typed_entities(index, "objecttype", spec.toggle_type)
        target_toggles &= _unary_entities(index, "toggleable")
        if not target_toggles:
            raise ValueError("ALFWorld verifier found no toggleable target light")
        held_objects = {row[1] for row in index.get("holds", set()) if len(row) == 2}
        toggled = _unary_entities(index, "istoggled")
        receptacle_locations = index.get("receptacleatlocation", set())
        agent_locations = {row[1] for row in index.get("atlocation", set()) if len(row) == 2}
        light_ready = any(toggle in toggled and any(placed_object == toggle and any(located_receptacle == receptacle and location in agent_locations for located_receptacle, location in receptacle_locations) for placed_object, receptacle in in_receptacle) for toggle in target_toggles)
        factors = {
            "target_held": float(bool(target_objects & held_objects)),
            "light_ready_at_agent": float(light_ready),
        }
        goal_satisfied = all(value == 1.0 for value in factors.values())
    else:
        if spec.receptacle_type is None:
            raise ValueError(f"{spec.task_type} requires receptacle_type")
        target_receptacles = _typed_entities(index, "receptacletype", spec.receptacle_type)
        placed_by_receptacle = {receptacle: {obj for obj, placed_receptacle in in_receptacle if placed_receptacle == receptacle and obj in target_objects} for receptacle in target_receptacles}
        placed_objects = set().union(*placed_by_receptacle.values())

        if spec.task_type == "pick_and_place_simple":
            factors = {"target_placed": float(bool(placed_objects))}
            goal_satisfied = factors["target_placed"] == 1.0
        elif spec.task_type == "pick_two_obj_and_place":
            max_placed = max((len(objects) for objects in placed_by_receptacle.values()), default=0)
            factors = {
                "one_target_placed": float(max_placed >= 1),
                "two_targets_placed": float(max_placed >= 2),
            }
            goal_satisfied = factors["two_targets_placed"] == 1.0
        else:
            state_specs = {
                "pick_clean_then_place_in_recep": ("cleanable", "isclean", "target_clean"),
                "pick_heat_then_place_in_recep": ("heatable", "ishot", "target_hot"),
                "pick_cool_then_place_in_recep": ("coolable", "iscool", "target_cool"),
            }
            capability_predicate, state_predicate, state_factor = state_specs[spec.task_type]
            eligible_objects = target_objects & _unary_entities(index, capability_predicate)
            if not eligible_objects:
                raise ValueError(f"ALFWorld verifier found no {capability_predicate} target object")
            state_objects = eligible_objects & _unary_entities(index, state_predicate)
            eligible_placed = eligible_objects & placed_objects
            joint_objects = state_objects & eligible_placed
            joint_factor = f"{state_factor}_and_placed"
            factors = {
                state_factor: float(bool(state_objects)),
                "target_placed": float(bool(eligible_placed)),
                joint_factor: float(bool(joint_objects)),
            }
            goal_satisfied = factors[joint_factor] == 1.0

    expected_factor_ids = ALFWORLD_TASK_FACTOR_IDS[spec.task_type]
    if tuple(factors) != expected_factor_ids:
        raise AssertionError("ALFWorld verifier factor ordering drifted from its schema")
    if bool(won_value) != bool(goal_satisfied):
        raise RuntimeError("ALFWorld goal-condition factors disagree with the official PDDL won signal")
    return factors
