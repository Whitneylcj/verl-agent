import numpy as np
import pytest
import torch

from recipe.exact.advantage import compute_exact_advantage, fit_delayed_alpha_from_traces
from recipe.exact.core_exact import DelayedAlphaController


class _Batch:
    def __init__(self, tensors, non_tensors):
        self.batch = tensors
        self.non_tensor_batch = non_tensors
        self.meta_info = {}


def _snapshot(checkpoint_id, values):
    return {
        "checkpoint_id": checkpoint_id,
        "factor_ids": ("a_value", "b_value"),
        "values": values,
        "read_sets": {"a_value": ("module:a",), "b_value": ("module:b",)},
        "channel_roles": {
            "a_value": "return_component",
            "b_value": "return_component",
        },
        "potential_weights": (1.0, 2.0),
        "source_revision": "fixture@1",
    }


def _test_batch():
    response_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 1],
            [1, 1, 0],
        ],
        dtype=torch.long,
    )
    return _Batch(
        tensors={
            "responses": torch.ones((4, 3), dtype=torch.long),
            "attention_mask": response_mask.clone(),
            "response_mask": response_mask.clone(),
        },
        non_tensors={
            "traj_uid": np.array(["t1", "t1", "t2", "t1"], dtype=object),
            "exact_step_id": np.array([1, 2, 1, 1]),
            "episode_rewards": np.array([3.0, 3.0, 2.0, 3.0]),
            "exact_padding": np.array([False, False, False, True]),
            "exact_probe_seconds": np.array([0.1, 0.2, 0.3, 99.0]),
            "exact_probe_count": np.array([2.0, 2.0, 2.0, 99.0]),
            "exact_factor_pre": np.array(
                [_snapshot(0, (0, 0)), _snapshot(1, (1, 0)), _snapshot(0, (0, 0)), _snapshot(0, (0, 0))],
                dtype=object,
            ),
            "exact_factor_post": np.array(
                [_snapshot(1, (1, 0)), _snapshot(2, (1, 1)), _snapshot(1, (0, 1)), _snapshot(1, (1, 0))],
                dtype=object,
            ),
            "exact_effect_schema": np.array(
                [
                    {"opaque": False, "descendant_factor_ids": ("a_value",), "certificate": "fixture"},
                    {"opaque": False, "descendant_factor_ids": ("b_value",), "certificate": "fixture"},
                    {"opaque": False, "descendant_factor_ids": ("b_value",), "certificate": "fixture"},
                    {"opaque": False, "descendant_factor_ids": ("a_value",), "certificate": "fixture"},
                ],
                dtype=object,
            ),
        },
    )


def test_exact_advantage_is_padding_safe_and_trajectory_normalized():
    data, metrics, traces = compute_exact_advantage(
        _test_batch(),
        config={
            "mode": "graph",
            "potential": {"weights": {"a_value": 1.0, "b_value": 2.0}},
        },
    )
    expected = torch.tensor(
        [
            [2.0, 2.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 4.0, 4.0],
            [0.0, 0.0, 0.0],
        ]
    )
    torch.testing.assert_close(data.batch["advantages"], expected)
    assert data.batch["response_mask"][3].sum() == 0
    assert data.batch["loss_mask"][3].sum() == 0
    assert data.meta_info["update_batch_mode"] == "full_rollout"
    assert data.meta_info["regularizer_scale"] == 2.0
    assert metrics["exact/trajectory_count"] == 2
    assert metrics["exact/padding_rows"] == 1
    assert metrics["exact/trajectory_scale"] == 2
    assert np.isclose(metrics["exact/probe_seconds"], 0.6)
    assert metrics["exact/verifier_snapshot_count"] == 6
    assert np.isclose(metrics["exact/probe_seconds_per_snapshot"], 0.1)
    assert len(traces) == 2
    assert traces[0]["atom_sum"] == traces[0]["episode_return"]
    assert set(traces[0]["channel_identities"]) == {"a_value", "b_value"}


def test_environment_native_potential_weights_are_default():
    data = _test_batch()

    data, metrics, _ = compute_exact_advantage(
        data,
        config={"mode": "graph"},
    )

    expected = torch.tensor(
        [
            [2.0, 2.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 4.0, 4.0],
            [0.0, 0.0, 0.0],
        ]
    )
    torch.testing.assert_close(data.batch["advantages"], expected)
    assert metrics["exact/closure_abs_mass_mean"] == 0.0
    assert metrics["exact/closure_abs_ratio_mean"] == 0.0
    assert metrics["exact/opaque_target_abs_ratio_mean"] == 0.0


def test_scaled_seq_mean_loss_equals_trajectory_mean_span_token_sum():
    pytest.importorskip("ray")
    from verl.trainer.ppo.core_algos import agg_loss

    data, _, _ = compute_exact_advantage(
        _test_batch(),
        config={
            "mode": "graph",
            "potential": {"weights": {"a_value": 1.0, "b_value": 2.0}},
        },
    )
    token_score_terms = torch.tensor(
        [
            [0.1, 0.2, 0.0],
            [0.3, 0.0, 0.0],
            [0.4, 0.5, 0.6],
            [9.0, 9.0, 0.0],
        ]
    )
    actual = agg_loss(
        loss_mat=token_score_terms * data.batch["advantages"],
        loss_mask=data.batch["response_mask"],
        loss_agg_mode="seq-mean-token-sum",
    )
    raw_credits = data.batch["advantages"] / 2.0
    expected = torch.sum(token_score_terms[:3] * raw_credits[:3]) / 2.0
    torch.testing.assert_close(actual, expected)


def test_full_batch_micro_accumulation_preserves_trajectory_mean():
    pytest.importorskip("ray")
    from verl.trainer.ppo.core_algos import agg_loss

    raw_credits = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    token_score_terms = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]])
    response_mask = torch.ones_like(raw_credits)
    trajectory_count = 2
    batch_size = len(raw_credits)
    micro_batch_size = 2
    accumulation_steps = batch_size // micro_batch_size
    scaled_credits = raw_credits * (batch_size / trajectory_count)

    accumulated = torch.tensor(0.0)
    for start in range(0, batch_size, micro_batch_size):
        end = start + micro_batch_size
        micro_loss = agg_loss(
            loss_mat=token_score_terms[start:end] * scaled_credits[start:end],
            loss_mask=response_mask[start:end],
            loss_agg_mode="seq-mean-token-sum",
        )
        accumulated = accumulated + micro_loss / accumulation_steps

    expected = torch.sum(token_score_terms * raw_credits) / trajectory_count
    torch.testing.assert_close(accumulated, expected)


def test_missing_schema_uses_all_to_all_fallback():
    data = _test_batch()
    data.non_tensor_batch["exact_effect_schema"][:3] = None
    data, metrics, _ = compute_exact_advantage(data, config={"mode": "graph"})
    assert metrics["exact/schema_fallback_rate"] == 1.0
    assert torch.all(data.batch["advantages"][:3][data.batch["response_mask"][:3].bool()] > 0)


def test_resource_graph_schema_compiles_model_reads_into_token_spans():
    data = _test_batch()
    data.non_tensor_batch["exact_effect_schema"] = np.array(
        [
            {
                "spans": [
                    {
                        "span_id": "selector",
                        "token_start": 0,
                        "token_end": 1,
                        "route_kind": "resource_graph",
                        "opaque": False,
                        "possible_write_set": ("module:a",),
                    },
                    {
                        "span_id": "arguments",
                        "token_start": 1,
                        "token_end": 2,
                        "route_kind": "resource_graph",
                        "opaque": False,
                        "possible_write_set": ("module:b",),
                        "control_parents": ("selector",),
                    },
                ]
            },
            {
                "route_kind": "resource_graph",
                "opaque": False,
                "possible_write_set": ("module:b",),
            },
            {
                "route_kind": "resource_graph",
                "opaque": False,
                "possible_write_set": ("module:b",),
            },
            {
                "route_kind": "resource_graph",
                "opaque": False,
                "possible_write_set": ("module:a",),
            },
        ],
        dtype=object,
    )
    data, metrics, traces = compute_exact_advantage(
        data,
        config={
            "mode": "graph",
            "potential": {"weights": {"a_value": 1.0, "b_value": 2.0}},
        },
    )

    torch.testing.assert_close(data.batch["advantages"][0], torch.tensor([6.0, 4.0, 0.0]))
    assert metrics["exact/resource_graph_span_rate"] == 1.0
    assert metrics["exact/unknown_factor_read_rate"] == 0.0
    t1_spans = next(trace["spans"] for trace in traces if trace["trajectory_id"] == "t1")
    assert len({span["span_id"] for span in t1_spans}) == len(t1_spans)


def test_checkpoint_discontinuity_fails_closed():
    data = _test_batch()
    data.non_tensor_batch["exact_factor_pre"][1] = _snapshot(1, (0, 0))
    try:
        compute_exact_advantage(data, config={"mode": "graph"})
    except ValueError as error:
        assert "discontinuity" in str(error)
    else:
        raise AssertionError("EXACT accepted a discontinuous verifier trajectory")


def test_sokoban_schema_rejects_unexplained_episode_return():
    data = _test_batch()
    for key in ("exact_factor_pre", "exact_factor_post"):
        for snapshot in data.non_tensor_batch[key]:
            snapshot["schema_version"] = "exact.sokoban.official_reward.v1"
    data.non_tensor_batch["episode_rewards"][:2] = 4.0

    with pytest.raises(AssertionError, match="do not reconstruct"):
        compute_exact_advantage(data, config={"mode": "graph"})


def test_graph_cv_uses_current_alpha_and_fits_only_next_alpha():
    controller = DelayedAlphaController(("default",), ridge=1e-9)
    data, metrics, traces = compute_exact_advantage(
        _test_batch(),
        config={"mode": "graph_cv"},
        alpha_by_bucket=controller.active,
    )
    assert metrics["exact/cv_alpha_active/default"] == 0.0
    pending = fit_delayed_alpha_from_traces(controller, traces, min_samples=1)
    assert pending is not None
    assert controller.active == {"default": 0.0}
    controller.advance()
    assert controller.active == pending
