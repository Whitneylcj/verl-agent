"""Side-effect-free environment factor probes used by EXACT rollouts."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import numpy as np


def _snapshot(
    factor_ids: Sequence[str],
    values: Sequence[float],
    read_sets: Mapping[str, Sequence[str]],
    schema_version: str,
    potential_weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    if len(factor_ids) != len(values):
        raise ValueError("factor IDs and values must be aligned")
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.ndim != 1 or not np.all(np.isfinite(values_array)):
        raise ValueError("probe factors must be a finite vector")
    snapshot = {
        "factor_ids": tuple(factor_ids),
        "values": tuple(values_array.tolist()),
        "read_sets": {key: tuple(value) for key, value in read_sets.items()},
        "schema_version": schema_version,
    }
    if potential_weights is not None:
        weights_array = np.asarray(potential_weights, dtype=np.float64)
        if weights_array.shape != values_array.shape or not np.all(np.isfinite(weights_array)):
            raise ValueError("probe potential weights must be finite and aligned with factors")
        snapshot["potential_weights"] = tuple(weights_array.tolist())
    return snapshot


SOKOBAN_REWARD_FACTOR_IDS = (
    "step_penalty",
    "box_on_target_reward",
    "box_off_target_penalty",
    "all_boxes_on_target_reward",
)


def sokoban_factor_snapshot(
    event_counts: Mapping[str, Any],
    reward_weights: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose cumulative events weighted exactly like gym-sokoban's reward."""

    expected = set(SOKOBAN_REWARD_FACTOR_IDS)
    for name, values in (("event counts", event_counts), ("reward weights", reward_weights)):
        missing = expected - set(values)
        extra = set(values) - expected
        if missing or extra:
            raise ValueError(f"Sokoban {name} must match the official reward factors; missing={sorted(missing)}, extra={sorted(extra)}")

    counts = np.asarray([event_counts[factor_id] for factor_id in SOKOBAN_REWARD_FACTOR_IDS], dtype=np.float64)
    if not np.all(np.isfinite(counts)) or np.any(counts < 0) or not np.all(counts == np.floor(counts)):
        raise ValueError("Sokoban official reward event counts must be finite non-negative integers")
    weights = np.asarray([reward_weights[factor_id] for factor_id in SOKOBAN_REWARD_FACTOR_IDS], dtype=np.float64)
    if not (weights[0] < 0 and weights[1] > 0 and weights[2] < 0 and weights[3] > 0):
        raise ValueError("Sokoban official reward weights have unexpected signs")

    read_sets = {
        "step_penalty": ("sokoban.interaction_count",),
        "box_on_target_reward": (
            "sokoban.box_positions",
            "sokoban.target_positions",
            "sokoban.reward_event_history",
        ),
        "box_off_target_penalty": (
            "sokoban.box_positions",
            "sokoban.target_positions",
            "sokoban.reward_event_history",
        ),
        "all_boxes_on_target_reward": (
            "sokoban.box_positions",
            "sokoban.target_positions",
            "sokoban.goal",
        ),
    }
    return _snapshot(
        SOKOBAN_REWARD_FACTOR_IDS,
        counts,
        read_sets,
        "exact.sokoban.official_reward.v1",
        potential_weights=weights,
    )


def _normalize_alfworld_entity(value: Any) -> str:
    return " ".join(str(value).lower().split())


def _alfworld_facts(raw_facts: Any) -> list[tuple[str, tuple[str, ...]]]:
    result = []
    for fact in raw_facts or ():
        if hasattr(fact, "name") and hasattr(fact, "arguments"):
            name = str(fact.name).lower()
            arguments = tuple(_normalize_alfworld_entity(getattr(argument, "name", argument)) for argument in fact.arguments)
        elif isinstance(fact, Mapping):
            name = str(fact["name"]).lower()
            raw_arguments = fact.get("arguments", fact.get("names", ()))
            arguments = tuple(_normalize_alfworld_entity(argument) for argument in raw_arguments)
        elif isinstance(fact, str):
            pieces = fact.split()
            name = pieces[0].lower()
            arguments = tuple(_normalize_alfworld_entity(piece) for piece in pieces[1:])
        else:
            continue
        result.append((name, arguments))
    return result


def alfworld_target_is_visible(
    info: Mapping[str, Any],
    task_params: Mapping[str, Any],
) -> bool:
    """Return whether the current observation exposes the task object."""

    object_target = _normalize_alfworld_entity(task_params.get("object_target", ""))
    if not object_target:
        return False
    observation = info.get("observation_text", "")
    if isinstance(observation, (list, tuple, np.ndarray)):
        observation = observation[0] if len(observation) else ""
    # The reset introduction repeats the target in the task description.  Only
    # inspect the scene portion so that this does not count as discovery.
    scene = _normalize_alfworld_entity(observation).split("your task is to:", 1)[0]
    target_pattern = rf"(?<!\w){re.escape(object_target)}(?: \d+)?(?!\w)"
    if re.search(target_pattern, scene):
        return True

    commands = info.get("admissible_commands", ())
    if isinstance(commands, str):
        commands = (commands,)
    command_pattern = re.compile(rf"^(?:take|examine) {re.escape(object_target)}(?: \d+)?(?: |$)")
    return any(command_pattern.search(_normalize_alfworld_entity(command)) for command in commands)


def alfworld_factor_snapshot(
    info: Mapping[str, Any],
    task_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile stable, verifier-only ALFWorld task progress factors."""

    if not task_params:
        won = float(bool(info.get("won", False)))
        return _snapshot(
            ("goal_satisfied",),
            (won,),
            {"goal_satisfied": ("alfworld.world_state", "alfworld.goal")},
            "exact.alfworld.goal.v1",
        )

    task_type = str(task_params["task_type"]).lower()
    object_target = _normalize_alfworld_entity(task_params.get("object_target", ""))
    parent_target = _normalize_alfworld_entity(task_params.get("parent_target", ""))
    toggle_target = _normalize_alfworld_entity(task_params.get("toggle_target", ""))
    facts = _alfworld_facts(info.get("facts", ()))

    def has_fact(predicate: str, *entities: str) -> bool:
        for name, arguments in facts:
            if name != predicate:
                continue
            if all(any(entity and entity in argument for argument in arguments) for entity in entities):
                return True
        return False

    held_objects = [arguments[-1] for name, arguments in facts if name == "holds" and arguments]
    holds_wrong_object = any(object_target not in held_object for held_object in held_objects)
    object_discovered = bool(info.get("exact.object_discovered", False)) or alfworld_target_is_visible(
        info,
        task_params,
    )
    values_by_id: dict[str, float] = {
        "object_discovered": float(object_discovered),
        "inventory_target_compatible": float(not holds_wrong_object),
        "object_acquired": float(has_fact("holds", object_target)),
    }
    if "clean" in task_type:
        values_by_id["object_cleaned"] = float(has_fact("isclean", object_target))
    if "heat" in task_type:
        values_by_id["object_heated"] = float(has_fact("ishot", object_target))
    if "cool" in task_type:
        values_by_id["object_cooled"] = float(has_fact("iscool", object_target))
    if "look_at" in task_type:
        values_by_id["light_toggled"] = float(has_fact("istoggled", toggle_target) or has_fact("ison", toggle_target))
    else:
        placed_objects = {arguments[0] for name, arguments in facts if name in {"inreceptacle", "inreceptacleobject"} and len(arguments) >= 2 and object_target in arguments[0] and parent_target in arguments[1]}
        values_by_id["object_placed_1"] = float(len(placed_objects) >= 1)
        if "pick_two" in task_type:
            values_by_id["object_placed_2"] = float(len(placed_objects) >= 2)
    values_by_id["goal_satisfied"] = float(bool(info.get("won", False)))

    read_sets = {factor_id: ("alfworld.world_facts", "alfworld.task_parameters") for factor_id in values_by_id}
    read_sets["object_discovered"] = (
        "alfworld.observation_history",
        "alfworld.task_parameters",
    )
    return _snapshot(
        tuple(values_by_id),
        tuple(values_by_id.values()),
        read_sets,
        f"exact.alfworld.{task_type}.v2",
    )


def webshop_factor_snapshot(score_components: Mapping[str, Any] | None) -> dict[str, Any]:
    """Expose the four components produced by WebShop's terminal scorer."""

    score_components = score_components or {}
    component_keys = ("r_type", "r_att", "r_option", "r_price")
    values = []
    for key in component_keys:
        value = score_components.get(key, 0.0)
        values.append(0.0 if value is None else float(value))
    read_sets = {
        "type_match": ("webshop.selected_product", "webshop.goal.type"),
        "attribute_match": ("webshop.selected_product", "webshop.goal.attributes"),
        "option_match": ("webshop.selected_options", "webshop.goal.options"),
        "price_match": ("webshop.selected_product", "webshop.goal.price"),
    }
    return _snapshot(
        ("type_match", "attribute_match", "option_match", "price_match"),
        values,
        read_sets,
        "exact.webshop.scorer.v1",
    )


def flatten_boolean_leaves(value: Any, prefix: str = "evaluation") -> dict[str, float]:
    """Flatten deterministic boolean test results from an AppWorld evaluation."""

    leaves: dict[str, float] = {}
    if isinstance(value, (bool, np.bool_)):
        leaves[prefix] = float(value)
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            child_prefix = f"{prefix}.{key}"
            leaves.update(flatten_boolean_leaves(value[key], child_prefix))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            leaves.update(flatten_boolean_leaves(item, f"{prefix}.{index}"))
    return leaves


def _appworld_requirement(entry: Any) -> str:
    if isinstance(entry, Mapping):
        for key in ("requirement", "name", "description"):
            if key in entry:
                return str(entry[key]).strip()
        stable_entry = {str(key): value for key, value in entry.items() if str(key).lower() not in {"error", "traceback", "stacktrace"}}
        return json.dumps(stable_entry, ensure_ascii=False, sort_keys=True, default=str)
    return str(entry).strip()


def _appworld_test_outcomes(evaluation: Mapping[str, Any]) -> dict[str, float]:
    from recipe.exact.appworld_schema import appworld_factor_id

    for container in (evaluation, *[value for value in evaluation.values() if isinstance(value, Mapping)]):
        if "passes" not in container or "failures" not in container:
            continue
        outcomes: dict[str, float] = {}
        requirements: dict[str, str] = {}
        for value, key in ((1.0, "passes"), (0.0, "failures")):
            entries = container[key]
            if not isinstance(entries, (list, tuple)):
                raise TypeError(f"AppWorld evaluation {key} must be a list")
            for entry in entries:
                requirement = _appworld_requirement(entry)
                if not requirement:
                    raise ValueError("AppWorld evaluation contains an empty requirement")
                factor_id = appworld_factor_id(requirement)
                if factor_id in requirements and requirements[factor_id] != requirement:
                    raise ValueError("AppWorld evaluation requirement hash collision")
                if factor_id in outcomes:
                    raise ValueError("AppWorld evaluation contains a duplicate requirement")
                requirements[factor_id] = requirement
                outcomes[factor_id] = value
        if outcomes:
            return outcomes
    return {}


def appworld_factor_snapshot(
    evaluation: Any,
    expected_factor_ids: Sequence[str] | None = None,
    read_sets: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Turn AppWorld's per-test evaluation booleans into a fixed factor vector."""

    if hasattr(evaluation, "to_dict"):
        evaluation = evaluation.to_dict()
    if not isinstance(evaluation, Mapping):
        raise TypeError("AppWorld evaluation must be a mapping or provide to_dict()")
    leaves = _appworld_test_outcomes(evaluation)
    if not leaves:
        leaves = flatten_boolean_leaves(evaluation)
    if not leaves:
        raise ValueError("AppWorld evaluation exposed no boolean test results")
    factor_ids = tuple(sorted(leaves)) if expected_factor_ids is None else tuple(expected_factor_ids)
    missing = set(factor_ids) - set(leaves)
    extra = set(leaves) - set(factor_ids)
    if missing or extra:
        raise ValueError(f"AppWorld evaluation schema changed; missing={sorted(missing)}, extra={sorted(extra)}")
    if read_sets is None:
        normalized_read_sets = {factor_id: (f"appworld.test:{factor_id}",) for factor_id in factor_ids}
        schema_version = "exact.appworld.tests.v1"
    else:
        missing_read_sets = set(factor_ids) - set(read_sets)
        extra_read_sets = set(read_sets) - set(factor_ids)
        if missing_read_sets or extra_read_sets:
            raise ValueError(f"AppWorld factor read-set schema mismatch; missing={sorted(missing_read_sets)}, extra={sorted(extra_read_sets)}")
        normalized_read_sets = {factor_id: tuple(read_sets[factor_id]) for factor_id in factor_ids}
        schema_version = "exact.appworld.tests.resource_graph.v2"
    return _snapshot(
        factor_ids,
        tuple(leaves[factor_id] for factor_id in factor_ids),
        normalized_read_sets,
        schema_version,
    )


def conservative_future_schema(snapshot: Mapping[str, Any], environment: str) -> dict[str, Any]:
    """Connect the action to every future factor; residual routing is compiler-enforced.

    This schema intentionally accounts for observation/history mediation through
    later policy actions. Environment-specific sparsity requires a stronger
    certificate and is not inferred from the action that happened to be sampled.
    """

    return {
        "opaque": False,
        "descendant_factor_ids": tuple(snapshot["factor_ids"]),
        "bucket": environment,
        "context_sources": (f"{environment}.observation_history",),
        "certificate": "all-future-factors-with-policy-context-mediation",
    }
