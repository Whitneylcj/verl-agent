"""Side-effect-free environment factor probes used by EXACT rollouts."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

GYM_SOKOBAN_SOURCE_REVISION = "mpSchrader/gym-sokoban@8e06e44e8bf3bb8bc73eeb1e7f0354508ce3fc89"
ALFWORLD_SOURCE_REVISION = "alfworld/alfworld@aaba6870f86c5be6a08a491f32a50b906227bc3e"
TEXTWORLD_SOURCE_REVISION = "microsoft/TextWorld@ebae03b2a65440f8baed46a885811719b1b948f2"
WEBSHOP_SOURCE_REVISION = "princeton-nlp/WebShop@64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"
APPWORLD_SOURCE_REVISION = "StonyBrookNLP/appworld@a072b7a86e7c1d5b1d7175659d750ebb9b79f10a"


def _snapshot(
    factor_ids: Sequence[str],
    values: Sequence[float],
    read_sets: Mapping[str, Sequence[str]],
    schema_version: str,
    channel_roles: Mapping[str, str],
    potential_weights: Sequence[float] | None = None,
    source_revision: str | None = None,
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
        "channel_roles": dict(channel_roles),
    }
    if set(channel_roles) != set(factor_ids):
        raise ValueError("probe channel_roles must exactly match factor IDs")
    if source_revision is not None:
        snapshot["source_revision"] = source_revision
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
        channel_roles={factor_id: "return_component" for factor_id in SOKOBAN_REWARD_FACTOR_IDS},
        potential_weights=weights,
        source_revision=GYM_SOKOBAN_SOURCE_REVISION,
    )


def alfworld_intermediate_reward_snapshot(cumulative_intermediate_reward: Any) -> dict[str, Any]:
    """Expose TextWorld's cumulative native intermediate reward.

    EXACT differences checkpoint potentials, so the one-step TextWorld signal is
    accumulated before it is recorded. This process-only channel has zero target
    mass; ALFWorld's existing ``10 * won`` training return remains opaque.
    """

    value = float(cumulative_intermediate_reward)
    if not np.isfinite(value):
        raise ValueError("ALFWorld cumulative intermediate_reward must be finite")
    channel_id = "textworld_intermediate_reward_cumulative"
    return _snapshot(
        (channel_id,),
        (value,),
        {channel_id: ("alfworld.textworld.quest_progression",)},
        "exact.alfworld.textworld.intermediate_reward.v1",
        channel_roles={channel_id: "process_verifier"},
        potential_weights=(1.0,),
        source_revision=f"{ALFWORLD_SOURCE_REVISION};{TEXTWORLD_SOURCE_REVISION}",
    )


def webshop_factor_snapshot(official_current_score: Any) -> dict[str, Any]:
    """Expose WebShop's official current-state score as one process channel."""

    value = float(official_current_score)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("WebShop official current score must be finite and in [0, 1]")
    channel_id = "webshop_official_current_score"
    return _snapshot(
        (channel_id,),
        (value,),
        {
            channel_id: (
                "webshop.session.asin",
                "webshop.session.options",
                "webshop.session.goal",
                "webshop.product.price",
                "webshop.product.catalog_fields_used_by_official_scorer",
            )
        },
        "exact.webshop.official_current_score.v1",
        channel_roles={channel_id: "process_verifier"},
        potential_weights=(1.0,),
        source_revision=WEBSHOP_SOURCE_REVISION,
    )


def compute_webshop_official_current_score(
    *,
    available_actions: Mapping[str, Any],
    session: Mapping[str, Any],
    product_item_dict: Mapping[str, Any],
    product_prices: Mapping[str, Any],
    scorer: Callable[..., Any],
) -> float:
    """Apply WebShop's official current-state score gate and scorer."""

    clickables = available_actions.get("clickables")
    if not isinstance(clickables, (list, tuple)):
        raise TypeError("WebShop available_actions.clickables must be a list")
    if "description" not in clickables:
        return 0.0
    asin = session.get("asin")
    if not asin:
        return 0.0
    score = float(
        scorer(
            product_item_dict[asin],
            session["goal"],
            price=product_prices.get(asin),
            options=session["options"],
        )
    )
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("WebShop official scorer returned a value outside [0, 1]")
    return score


def _appworld_requirement(entry: Any) -> str:
    if not isinstance(entry, Mapping) or "requirement" not in entry:
        raise TypeError("AppWorld pass/failure entries must carry a requirement field")
    requirement = str(entry["requirement"]).strip()
    if not requirement:
        raise ValueError("AppWorld evaluation contains an empty requirement")
    return requirement


def _appworld_test_outcomes(evaluation: Mapping[str, Any]) -> dict[str, float]:
    from recipe.exact.appworld_schema import appworld_factor_id

    required_keys = {"passes", "failures", "num_tests", "success"}
    missing_keys = required_keys - set(evaluation)
    if missing_keys:
        raise ValueError(f"AppWorld TestTracker output is missing keys: {sorted(missing_keys)}")
    num_tests = evaluation["num_tests"]
    if isinstance(num_tests, (bool, np.bool_)) or not isinstance(num_tests, (int, np.integer)):
        raise TypeError("AppWorld num_tests must be an integer")
    if int(num_tests) <= 0:
        raise ValueError("AppWorld num_tests must be positive")
    success = evaluation["success"]
    if not isinstance(success, (bool, np.bool_)):
        raise TypeError("AppWorld success must be boolean")

    outcomes: dict[str, float] = {}
    requirements: dict[str, str] = {}
    for value, key in ((1.0, "passes"), (0.0, "failures")):
        entries = evaluation[key]
        if not isinstance(entries, (list, tuple)):
            raise TypeError(f"AppWorld evaluation {key} must be a list")
        for entry in entries:
            requirement = _appworld_requirement(entry)
            factor_id = appworld_factor_id(requirement)
            if factor_id in requirements and requirements[factor_id] != requirement:
                raise ValueError("AppWorld evaluation requirement hash collision")
            if factor_id in outcomes:
                raise ValueError("AppWorld evaluation contains a duplicate requirement")
            requirements[factor_id] = requirement
            outcomes[factor_id] = value
    if len(outcomes) != int(num_tests):
        raise ValueError("AppWorld pass/failure entries are not exhaustive for num_tests")
    expected_success = all(value == 1.0 for value in outcomes.values())
    if bool(success) != expected_success:
        raise ValueError("AppWorld success is inconsistent with pass/failure outcomes")
    return outcomes


def appworld_factor_snapshot(
    evaluation: Any,
    expected_factor_ids: Sequence[str] | None = None,
    read_sets: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Turn AppWorld's per-test evaluation booleans into a fixed factor vector."""

    if hasattr(evaluation, "to_dict"):
        evaluation = evaluation.to_dict(stats_only=False)
    if not isinstance(evaluation, Mapping):
        raise TypeError("AppWorld evaluation must be a mapping or provide to_dict()")
    leaves = _appworld_test_outcomes(evaluation)
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
        channel_roles={factor_id: "process_verifier" for factor_id in factor_ids},
        potential_weights=np.full(len(factor_ids), 1.0 / len(factor_ids)),
        source_revision=APPWORLD_SOURCE_REVISION,
    )


def conservative_future_schema(snapshot: Mapping[str, Any], environment: str) -> dict[str, Any]:
    """Connect the action to every future channel; opaque-target routing is explicit.

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
