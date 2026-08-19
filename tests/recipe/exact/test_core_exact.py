import itertools

import numpy as np
import pytest

from recipe.exact.core_exact import (
    DelayedAlphaController,
    IdentityPotential,
    build_scoped_conserved_atoms,
    compute_exact_credits,
)
from recipe.exact.credit_spec import EffectSpan, FactorSnapshot, SpanRoute, snapshots_from_values


def _two_action_fixture(a: int, b: int):
    snapshots = snapshots_from_values(
        factor_ids=("a_value", "b_value"),
        checkpoint_values=((0, 0), (a, 0), (a, b)),
        read_sets={"a_value": ("module:a",), "b_value": ("module:b",)},
        channel_roles={"a_value": "return_component", "b_value": "return_component"},
        potential_weights=(1.0, 2.0),
    )
    potential = IdentityPotential(factor_ids=("a_value", "b_value"), weights=np.array([1.0, 2.0]))
    conserved = build_scoped_conserved_atoms(
        snapshots,
        episode_return=float(a + 2 * b),
        potential=potential,
    )
    routes = (
        SpanRoute(
            span=EffectSpan("span:a", step_id=1, token_start=0, token_end=1),
            descendant_atom_ids=(
                "delta:1:a_value",
                "delta:2:a_value",
                "closure:a_value",
            ),
            soundness_certificate="enumerated-independent-modules",
        ),
        SpanRoute(
            span=EffectSpan("span:b", step_id=2, token_start=1, token_end=2),
            descendant_atom_ids=("delta:2:b_value", "closure:b_value"),
            soundness_certificate="enumerated-independent-modules",
        ),
    )
    return conserved, routes


def test_pathwise_and_per_channel_conservation_with_graph_credit():
    conserved, routes = _two_action_fixture(a=1, b=0)
    assert conserved.conservation_error < 1e-12
    assert max(conserved.channel_conservation_errors.values()) < 1e-12
    assert conserved.channel_target_masses == {"a_value": 1.0, "b_value": 0.0}
    assert sum(atom.value for atom in conserved.atoms) == 1.0

    result = compute_exact_credits(conserved, routes, mode="graph")
    assert [credit.credit for credit in result.span_credits] == [1.0, 0.0]
    assert result.cone_density == 0.5
    assert result.closure_route_density == 0.5


def test_temporal_credit_equals_future_return():
    conserved, routes = _two_action_fixture(a=1, b=1)
    result = compute_exact_credits(conserved, routes, mode="temporal")
    assert [credit.credit for credit in result.span_credits] == [3.0, 2.0]


def test_exact_gradient_matches_enumeration_and_false_edges_are_safe():
    p_a, p_b = 0.35, 0.6
    exact_gradient = np.zeros(2)
    conservative_gradient = np.zeros(2)
    outcome_gradient = np.zeros(2)

    for a, b in itertools.product((0, 1), repeat=2):
        probability = (p_a if a else 1 - p_a) * (p_b if b else 1 - p_b)
        scores = np.array([a - p_a, b - p_b])
        episode_return = a + 2 * b
        conserved, routes = _two_action_fixture(a, b)
        result = compute_exact_credits(conserved, routes, mode="graph")
        exact_gradient += probability * scores * np.array([credit.credit for credit in result.span_credits])

        false_edge_routes = (
            SpanRoute(
                routes[0].span,
                (
                    "delta:1:a_value",
                    "delta:2:a_value",
                    "closure:a_value",
                    "delta:2:b_value",
                    "closure:b_value",
                ),
                "conservative-false-edge",
            ),
            routes[1],
        )
        conservative = compute_exact_credits(conserved, false_edge_routes, mode="graph")
        conservative_gradient += probability * scores * np.array([credit.credit for credit in conservative.span_credits])
        outcome_gradient += probability * scores * episode_return

    expected = np.array([p_a * (1 - p_a), 2 * p_b * (1 - p_b)])
    np.testing.assert_allclose(exact_gradient, expected, atol=1e-12)
    np.testing.assert_allclose(conservative_gradient, expected, atol=1e-12)
    np.testing.assert_allclose(outcome_gradient, expected, atol=1e-12)


def test_deleting_true_edge_produces_bias():
    p_a = 0.35
    deleted_edge_gradient = 0.0
    for a, b in itertools.product((0, 1), repeat=2):
        probability = (p_a if a else 1 - p_a) * 0.5
        conserved, routes = _two_action_fixture(a, b)
        broken_routes = (SpanRoute(routes[0].span, (), "intentionally-unsound"), routes[1])
        result = compute_exact_credits(conserved, broken_routes, mode="graph")
        deleted_edge_gradient += probability * (a - p_a) * result.span_credits[0].credit
    assert deleted_edge_gradient == 0.0
    assert not np.isclose(deleted_edge_gradient, p_a * (1 - p_a))


def test_process_verifier_closure_is_local_and_opaque_target_carries_return():
    snapshots = snapshots_from_values(
        ("progress",),
        ((0,), (1,), (0.25,)),
        read_sets={"progress": ("verifier:progress",)},
        channel_roles={"progress": "process_verifier"},
    )
    potential = IdentityPotential(("progress",), np.array([2.0]))
    conserved = build_scoped_conserved_atoms(snapshots, episode_return=1.0, potential=potential)

    assert conserved.channel_target_masses == {"progress": 0.0}
    assert conserved.closure_atoms[0].read_set == ("verifier:progress",)
    assert conserved.closure_atoms[0].value == -0.5
    assert conserved.opaque_target_atom.value == 1.0
    assert np.isclose(sum(atom.value for atom in conserved.atoms), 1.0)


def test_return_components_reject_non_native_potential_weights():
    snapshots = snapshots_from_values(
        ("reward",),
        ((0,), (1,)),
        channel_roles={"reward": "return_component"},
        potential_weights=(2.0,),
    )
    with pytest.raises(ValueError, match="exact native weights"):
        build_scoped_conserved_atoms(
            snapshots,
            episode_return=2.0,
            potential=IdentityPotential(("reward",), np.array([1.0])),
        )


def test_snapshot_metadata_must_remain_stable():
    before = FactorSnapshot(
        checkpoint_id=0,
        factor_ids=("progress",),
        values=np.array([0.0]),
        channel_roles={"progress": "process_verifier"},
    )
    after = FactorSnapshot(
        checkpoint_id=1,
        factor_ids=("progress",),
        values=np.array([1.0]),
        channel_roles={"progress": "return_component"},
        potential_weights=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="channel roles changed"):
        build_scoped_conserved_atoms(
            (before, after),
            episode_return=1.0,
            potential=IdentityPotential(("progress",), np.array([1.0])),
        )


def test_opaque_route_receives_every_atom():
    conserved, _ = _two_action_fixture(a=1, b=1)
    opaque_route = SpanRoute(
        span=EffectSpan("opaque", step_id=1, token_start=0, token_end=2, opaque=True),
        descendant_atom_ids=(),
    )
    result = compute_exact_credits(conserved, (opaque_route,), mode="graph")
    assert result.span_credits[0].credit == conserved.episode_return
    assert result.cone_density == 1.0


def test_delayed_alpha_only_activates_after_advance():
    controller = DelayedAlphaController(("tool",), ridge=1e-9)
    hard = np.array([[1.0], [-1.0], [2.0], [-2.0]])
    controls = (-hard)[:, None, :]
    pending = controller.fit_pending(hard, controls)
    assert controller.active == {"tool": 0.0}
    assert np.isclose(pending["tool"], 1.0)
    active = controller.advance()
    assert np.isclose(active["tool"], 1.0)


def test_delayed_alpha_checkpoint_round_trip_and_clipping():
    controller = DelayedAlphaController(("tool",), ridge=1e-9, max_abs_alpha=0.25)
    hard = np.array([[10.0], [-10.0]])
    controls = (-hard)[:, None, :]
    pending = controller.fit_pending(hard, controls)
    assert pending == {"tool": 0.25}
    controller.advance()

    restored = DelayedAlphaController(("tool",), ridge=1e-9, max_abs_alpha=0.25)
    restored.load_state_dict(controller.state_dict())
    assert restored.active == controller.active
