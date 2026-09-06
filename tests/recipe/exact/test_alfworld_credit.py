import copy
import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from recipe.exact.advantage import compute_exact_advantage
from recipe.exact.alfworld_adapter import resolve_effect_schema
from tests.recipe.exact.alfworld_fixtures import TASKS, CharacterTokenizer, execute, response, session


def trajectory_batch(actions, task=TASKS[1], horizon=20, padding=False, guard=True):
    current = session(task, horizon=horizon, guard=guard)
    pre, post, schemas, lengths = [], [], [], []
    for op, args, commit in actions:
        pre.append(current.record)
        text = response(op, *args, commit=commit)
        lengths.append(len(text))
        schemas.append(resolve_effect_schema(current.record["prefix_registry"], list(map(ord, text)), CharacterTokenizer()))
        execute(current, op, *args, commit=commit)
        post.append(current.record)
        if current.state["won"]:
            break
    n = len(pre)
    non_tensors = {
        "traj_uid": ["t"] * n,
        "exact_step_id": list(range(1, n + 1)),
        "episode_rewards": [10.0 * current.state["won"]] * n,
        "exact_factor_pre": pre,
        "exact_factor_post": post,
        "exact_effect_schema": schemas,
        "exact_padding": [False] * n,
    }
    if padding:
        for key, values in non_tensors.items():
            values.append(values[0] if key != "exact_padding" else True)
        lengths.append(lengths[0])
    mask = torch.tensor([[1] * length + [0] * (max(lengths) - length) for length in lengths])
    return SimpleNamespace(batch={"responses": torch.ones_like(mask), "attention_mask": mask.clone(), "response_mask": mask}, non_tensor_batch={k: np.array(v, dtype=object) for k, v in non_tensors.items()}, meta_info={})


CLEAN_AND_LOCK = [("take", ("apple1", "table"), False), ("goto", ("sink",), False), ("clean", ("apple1", "sink"), False), ("goto", ("table",), False), ("put", ("apple1", "table"), True), ("take", ("apple1", "table"), False), ("inventory", (), False)]


def test_fixed_horizon_nonzero_closures_padding_and_baseline_equality():
    batch = trajectory_batch(CLEAN_AND_LOCK, padding=True)
    results = {}
    for mode in ("graph", "temporal", "prefix_baseline"):
        data, metrics, traces = compute_exact_advantage(copy.deepcopy(batch), {"mode": mode})
        results[mode] = data.batch["advantages"]
        assert metrics["exact/conservation_error_max"] < 1e-10
        assert not data.batch["advantages"][-1].any()
        trace = traces[0]
        assert trace["atom_sum"] == pytest.approx(0)
        comparison = trace["alfworld_comparison"]["metrics"]
        assert comparison["delta_pair_reduction"] > 0
        assert comparison["closure_pair_reduction"] > 0
        assert comparison["nonzero_mass_removed"] > 0
        assert comparison["credit_difference_abs"] > 0
        assert comparison["prefix_baseline_error_max"] < 1e-10
    torch.testing.assert_close(results["graph"], results["prefix_baseline"])
    assert not torch.equal(results["graph"], results["temporal"])
    # Extending after termination creates no extra policy scores or sparsity gains.
    _, _, longer = compute_exact_advantage(trajectory_batch(CLEAN_AND_LOCK, horizon=40), {"mode": "graph"})
    assert longer[0]["alfworld_comparison"]["metrics"] == trace["alfworld_comparison"]["metrics"]


def test_early_success_and_timeout_keep_official_target():
    success = [("take", ("apple1", "table"), False), ("goto", ("fridge",), False), ("open", ("fridge",), False), ("put", ("apple1", "fridge"), True)]
    for actions, expected, horizon in ((success, 10, 20), ([("inventory", (), False)] * 3, 0, 3)):
        _, _, traces = compute_exact_advantage(trajectory_batch(actions, task=TASKS[0], horizon=horizon), {"mode": "graph"})
        assert traces[0]["episode_return"] == expected
        assert traces[0]["atom_sum"] == pytest.approx(expected)
        assert traces[0]["alfworld_comparison"]["metrics"]["prefix_baseline_error_max"] < 1e-10


def test_placement_guard_removes_nonzero_temperature_closure_transitively():
    actions = [("take", ("apple1", "table"), False), ("goto", ("micro",), False), ("heat", ("apple1", "micro"), False), ("goto", ("table",), False), ("put", ("apple1", "table"), True), ("inventory", (), False)]
    traces = {}
    for guard in (False, True):
        _, _, result = compute_exact_advantage(trajectory_batch(actions, task=TASKS[2], guard=guard), {"mode": "graph"})
        traces[guard] = result[0]
    # Both physical paths heat the same apple and end at the same wrong table.
    # The placement guard alone makes the heat action permanently unreachable.
    assert traces[False]["atoms"] == traces[True]["atoms"]
    assert traces[True]["spans"][-1]["credit"] == pytest.approx(0)
    assert traces[False]["spans"][-1]["credit"] < 0
    assert traces[True]["alfworld_comparison"]["metrics"]["closure_nonzero_abs_mass_removed"] > 0


@pytest.mark.parametrize("corruption", ["source", "value", "invariant", "missing"])
def test_corrupt_or_missing_evidence_stops_update(corruption):
    batch = trajectory_batch(CLEAN_AND_LOCK)
    schema = batch.non_tensor_batch["exact_effect_schema"][0]
    if corruption == "source":
        schema["source_revision"] = "unknown"
    elif corruption == "value":
        schema["pre_values"] = (99,) * len(schema["pre_values"])
    elif corruption == "invariant":
        schema["spans"][0]["certificate"]["invariant_values"] = (("holds(a,apple1)", 0),)
    else:
        batch.non_tensor_batch["exact_effect_schema"][0] = None
    with pytest.raises(ValueError):
        compute_exact_advantage(batch, {"mode": "graph"})


def test_exhaustive_on_policy_expected_gradient_matches_official_return():
    # Three stochastic decisions, shared Bernoulli logit, complete enumeration.
    # Clean is a natural commit; optional premature placement creates a guard.
    # Later decisions depend on earlier state; invalid actions remain outcomes.
    p = 0.37
    means = dict.fromkeys(("graph", "temporal", "prefix_baseline", "official"), 0.0)
    for bits in itertools.product((0, 1), repeat=3):
        actions = [
            ("take", ("apple1", "table"), False),
            ("goto", ("sink",), False),
            ("clean" if bits[0] else "inventory", ("apple1", "sink") if bits[0] else (), False),
            ("goto", ("table",), False),
            ("put" if bits[1] else "inventory", ("apple1", "table") if bits[1] else (), True if bits[1] else False),
            ("goto", ("fridge",), False),
            ("open", ("fridge",), False),
            ("put" if bits[2] else "inventory", ("apple1", "fridge") if bits[2] else (), False),
        ]
        batch = trajectory_batch(actions)
        probability = np.prod([p if bit else 1 - p for bit in bits])
        rows = (2, 4, 7)
        for mode in ("graph", "temporal", "prefix_baseline"):
            _, _, traces = compute_exact_advantage(copy.deepcopy(batch), {"mode": mode})
            # The first differing op character carries the Bernoulli score;
            # shared JSON framing and subsequent serialization are deterministic.
            offset = len('{"op":"')
            span_ids = {f"row:{row}:{record['span_id']}" for row, schema in enumerate(traces[0]["prefix_schemas"]) for record in schema["spans"] if record["token_start"] <= offset < record["token_end"]}
            raw = [span["credit"] for span in traces[0]["spans"] if span["span_id"] in span_ids]
            means[mode] += probability * sum((bit - p) * raw[row] for bit, row in zip(bits, rows))
        means["official"] += probability * sum(bit - p for bit in bits) * batch.non_tensor_batch["episode_rewards"][0]
    assert abs(means["official"]) > 0.1
    assert np.max(np.abs(np.array(list(means.values())) - means["official"])) < 1e-10
