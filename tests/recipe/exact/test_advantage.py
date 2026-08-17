import numpy as np
import torch

from recipe.exact.advantage import compute_exact_advantage, fit_delayed_alpha_from_traces
from recipe.exact.core_exact import DelayedAlphaController


class _Batch:
    def __init__(self, tensors, non_tensors):
        self.batch = tensors
        self.non_tensor_batch = non_tensors


def _snapshot(checkpoint_id, values):
    return {
        "checkpoint_id": checkpoint_id,
        "factor_ids": ("a_value", "b_value"),
        "values": values,
        "read_sets": {"a_value": ("module:a",), "b_value": ("module:b",)},
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
            "force_residual_descendant": False,
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
    assert metrics["exact/trajectory_count"] == 2
    assert metrics["exact/padding_rows"] == 1
    assert metrics["exact/trajectory_scale"] == 2
    assert len(traces) == 2


def test_missing_schema_uses_all_to_all_fallback():
    data = _test_batch()
    data.non_tensor_batch["exact_effect_schema"][:3] = None
    data, metrics, _ = compute_exact_advantage(data, config={"mode": "graph"})
    assert metrics["exact/schema_fallback_rate"] == 1.0
    assert torch.all(data.batch["advantages"][:3][data.batch["response_mask"][:3].bool()] > 0)


def test_checkpoint_discontinuity_fails_closed():
    data = _test_batch()
    data.non_tensor_batch["exact_factor_pre"][1] = _snapshot(1, (0, 0))
    try:
        compute_exact_advantage(data, config={"mode": "graph"})
    except ValueError as error:
        assert "discontinuity" in str(error)
    else:
        raise AssertionError("EXACT accepted a discontinuous verifier trajectory")


def test_graph_cv_uses_current_alpha_and_fits_only_next_alpha():
    controller = DelayedAlphaController(("default",), ridge=1e-9)
    data, metrics, traces = compute_exact_advantage(
        _test_batch(),
        config={"mode": "graph_cv", "force_residual_descendant": False},
        alpha_by_bucket=controller.active,
    )
    assert metrics["exact/cv_alpha_active/default"] == 0.0
    pending = fit_delayed_alpha_from_traces(controller, traces, min_samples=1)
    assert pending is not None
    assert controller.active == {"default": 0.0}
    controller.advance()
    assert controller.active == pending
