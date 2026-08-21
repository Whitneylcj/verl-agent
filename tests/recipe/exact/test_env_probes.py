import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from recipe.exact.env_probes import (
    ALFWORLD_SOURCE_REVISION,
    APPWORLD_SOURCE_REVISION,
    GYM_SOKOBAN_SOURCE_REVISION,
    TEXTWORLD_SOURCE_REVISION,
    WEBSHOP_SOURCE_REVISION,
    alfworld_intermediate_reward_snapshot,
    appworld_factor_snapshot,
    compute_webshop_official_current_score,
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
    assert set(snapshot["channel_roles"].values()) == {"return_component"}
    assert snapshot["source_revision"] == GYM_SOKOBAN_SOURCE_REVISION


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


def test_alfworld_uses_one_native_cumulative_process_channel():
    snapshot = alfworld_intermediate_reward_snapshot(2.0)
    channel_id = "textworld_intermediate_reward_cumulative"
    assert snapshot["factor_ids"] == (channel_id,)
    assert snapshot["values"] == (2.0,)
    assert snapshot["channel_roles"] == {channel_id: "process_verifier"}
    assert snapshot["potential_weights"] == (1.0,)
    assert snapshot["read_sets"] == {
        channel_id: ("alfworld.textworld.quest_progression",)
    }
    assert snapshot["schema_version"] == "exact.alfworld.textworld.intermediate_reward.v1"
    assert snapshot["source_revision"] == (f"{ALFWORLD_SOURCE_REVISION};{TEXTWORLD_SOURCE_REVISION}")


def test_alfworld_snapshot_rejects_nonfinite_cumulative_value():
    with pytest.raises(ValueError, match="must be finite"):
        alfworld_intermediate_reward_snapshot(float("nan"))


def test_webshop_uses_one_official_current_score_channel():
    snapshot = webshop_factor_snapshot(0.625)
    assert snapshot["factor_ids"] == ("webshop_official_current_score",)
    assert snapshot["values"] == (0.625,)
    assert snapshot["channel_roles"] == {"webshop_official_current_score": "process_verifier"}
    assert snapshot["source_revision"] == WEBSHOP_SOURCE_REVISION
    for invalid in (-0.01, 1.01, float("nan")):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            webshop_factor_snapshot(invalid)


def test_webshop_current_score_matches_direct_official_scorer_call():
    calls = []

    def scorer(product, goal, *, price, options):
        calls.append((product, goal, price, options))
        return 0.75

    product = {"Title": "item"}
    goal = {"instruction_text": "buy item"}
    options = {"color": "blue"}
    score = compute_webshop_official_current_score(
        available_actions={"clickables": ["description", "buy now"]},
        session={"asin": "A1", "goal": goal, "options": options},
        product_item_dict={"A1": product},
        product_prices={"A1": 9.99},
        scorer=scorer,
    )

    assert score == 0.75
    assert calls == [(product, goal, 9.99, options)]


def test_webshop_current_score_is_zero_off_item_page_without_scorer_call():
    def unexpected_scorer(*args, **kwargs):
        raise AssertionError((args, kwargs))

    score = compute_webshop_official_current_score(
        available_actions={"clickables": ["search"]},
        session={"asin": None, "goal": {}, "options": {}},
        product_item_dict={},
        product_prices={},
        scorer=unexpected_scorer,
    )
    assert score == 0.0


def test_webshop_score_tracks_official_options_without_mutating_session():
    session = {
        "asin": "A1",
        "goal": {"desired_color": "blue"},
        "options": {"color": "red"},
        "actions": {"options": 1},
        "done": False,
        "reward": 0.0,
    }
    original = copy.deepcopy(session)

    def scorer(product, goal, *, price, options):
        del product, price
        return float(options["color"] == goal["desired_color"])

    kwargs = {
        "available_actions": {"clickables": ["description"]},
        "session": session,
        "product_item_dict": {"A1": {"Title": "item"}},
        "product_prices": {"A1": 1.0},
        "scorer": scorer,
    }
    assert compute_webshop_official_current_score(**kwargs) == 0.0
    assert compute_webshop_official_current_score(**kwargs) == 0.0
    assert session == original

    session["options"]["color"] = "blue"
    assert compute_webshop_official_current_score(**kwargs) == 1.0


@pytest.mark.parametrize(("official_task_score", "expected_training_reward", "expected_won"), ((0.75, 0.0, False), (1.0, 10.0, True)))
def test_webshop_worker_preserves_raw_task_score_before_binarizing_reward(
    monkeypatch,
    official_task_score,
    expected_training_reward,
    expected_won,
):
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "gym", SimpleNamespace(Env=object))
    module_path = Path(__file__).parents[3] / "agent_system/environments/env_package/webshop/envs.py"
    spec = importlib.util.spec_from_file_location("webshop_envs_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeWebshopEnv:
        session = "session-1"
        server = SimpleNamespace(user_sessions={"session-1": {"verbose_info": {"r_att": 0.5}}})

        def step(self, action):
            assert action == "click[buy now]"
            return "terminal observation", official_task_score, True, {}

        def get_available_actions(self):
            return {"has_search_bar": True, "clickables": []}

        def official_current_score(self, available_actions):
            assert available_actions == self.get_available_actions()
            return 0.0

    worker = object.__new__(module.WebshopWorker)
    worker.env = FakeWebshopEnv()
    worker._exact_official_current_score = 0.5

    _, reward, done, info = worker.step("click[buy now]")

    assert done
    assert info["task_score"] == official_task_score
    assert reward == expected_training_reward
    assert info["won"] is expected_won
    assert worker._exact_official_current_score == 0.0


def _evaluation(*, passes, failures, success):
    return {
        "num_tests": len(passes) + len(failures),
        "success": success,
        "passes": [{"requirement": value} for value in passes],
        "failures": [{"requirement": value} for value in failures],
    }


def test_appworld_per_requirement_schema_is_fixed_and_normalized():
    evaluation = _evaluation(
        passes=("created the playlist",),
        failures=("shared the playlist",),
        success=False,
    )
    first = appworld_factor_snapshot(evaluation)
    second = appworld_factor_snapshot(evaluation, expected_factor_ids=first["factor_ids"])
    assert first == second
    assert set(first["channel_roles"].values()) == {"process_verifier"}
    assert first["potential_weights"] == (0.5, 0.5)
    assert first["source_revision"] == APPWORLD_SOURCE_REVISION

    progressed = _evaluation(
        passes=("created the playlist", "shared the playlist"),
        failures=(),
        success=True,
    )
    progressed_snapshot = appworld_factor_snapshot(
        progressed,
        expected_factor_ids=first["factor_ids"],
    )
    assert progressed_snapshot["values"] == (1.0, 1.0)


class _Tracker:
    def __init__(self, payload):
        self.payload = payload
        self.stats_only = None

    def to_dict(self, *, stats_only):
        self.stats_only = stats_only
        return self.payload


def test_appworld_requests_full_testtracker_requirements():
    tracker = _Tracker(_evaluation(passes=("done",), failures=(), success=True))
    snapshot = appworld_factor_snapshot(tracker)
    assert tracker.stats_only is False
    assert snapshot["values"] == (1.0,)


@pytest.mark.parametrize(
    "evaluation,match",
    (
        ({"passes": [], "failures": []}, "missing keys"),
        (
            {"num_tests": 1, "success": True, "passes": [], "failures": []},
            "not exhaustive",
        ),
        (
            {
                "num_tests": 1,
                "success": False,
                "passes": [{"name": "not an official requirement"}],
                "failures": [],
            },
            "requirement field",
        ),
        (
            _evaluation(passes=("done",), failures=(), success=False),
            "inconsistent",
        ),
    ),
)
def test_appworld_malformed_or_incomplete_tracker_output_fails_closed(evaluation, match):
    with pytest.raises((TypeError, ValueError), match=match):
        appworld_factor_snapshot(evaluation)


def test_appworld_rejects_duplicate_requirements():
    evaluation = {
        "num_tests": 2,
        "success": False,
        "passes": [{"requirement": "same"}],
        "failures": [{"requirement": "same"}],
    }
    with pytest.raises(ValueError, match="duplicate"):
        appworld_factor_snapshot(evaluation)


def test_default_effect_schema_keeps_every_future_channel():
    snapshot = webshop_factor_snapshot(0.0)
    schema = conservative_future_schema(snapshot, "webshop")
    assert schema["descendant_factor_ids"] == snapshot["factor_ids"]
    assert schema["opaque"] is False
