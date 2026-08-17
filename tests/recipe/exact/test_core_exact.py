import itertools

import numpy as np

from recipe.exact.core_exact import (
    DelayedAlphaController,
    IdentityPotential,
    build_conserved_atoms,
    compute_exact_credits,
)
from recipe.exact.credit_spec import EffectSpan, SpanRoute, snapshots_from_values


def _two_action_fixture(a: int, b: int):
    snapshots = snapshots_from_values(
        factor_ids=("a_value", "b_value"),
        checkpoint_values=((0, 0), (a, 0), (a, b)),
        read_sets={"a_value": ("module:a",), "b_value": ("module:b",)},
    )
    potential = IdentityPotential(factor_ids=("a_value", "b_value"), weights=np.array([1.0, 2.0]))
    conserved = build_conserved_atoms(snapshots, episode_return=float(a + 2 * b), potential=potential)
    routes = (
        SpanRoute(
            span=EffectSpan("span:a", step_id=1, token_start=0, token_end=1),
            descendant_atom_ids=("factor:1:a_value",),
            soundness_certificate="enumerated-independent-modules",
        ),
        SpanRoute(
            span=EffectSpan("span:b", step_id=2, token_start=1, token_end=2),
            descendant_atom_ids=("factor:2:b_value",),
            soundness_certificate="enumerated-independent-modules",
        ),
    )
    return conserved, routes


def test_pathwise_conservation_and_graph_credit():
    conserved, routes = _two_action_fixture(a=1, b=0)
    assert conserved.conservation_error < 1e-12
    assert sum(atom.value for atom in conserved.atoms) == 1.0

    result = compute_exact_credits(conserved, routes, mode="graph", force_residual_descendant=False)
    assert [credit.credit for credit in result.span_credits] == [1.0, 0.0]
    # The fixed schema keeps zero-valued factor atoms, so there are five atoms:
    # two factors at each of two checkpoints plus the residual.
    assert result.cone_density == 1 / 5


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
        result = compute_exact_credits(conserved, routes, mode="graph", force_residual_descendant=False)
        exact_gradient += probability * scores * np.array([credit.credit for credit in result.span_credits])

        false_edge_routes = (
            SpanRoute(routes[0].span, ("factor:1:a_value", "factor:2:b_value"), "conservative-false-edge"),
            routes[1],
        )
        conservative = compute_exact_credits(
            conserved,
            false_edge_routes,
            mode="graph",
            force_residual_descendant=False,
        )
        conservative_gradient += probability * scores * np.array(
            [credit.credit for credit in conservative.span_credits]
        )
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
        result = compute_exact_credits(
            conserved,
            broken_routes,
            mode="graph",
            force_residual_descendant=False,
        )
        deleted_edge_gradient += probability * (a - p_a) * result.span_credits[0].credit
    assert deleted_edge_gradient == 0.0
    assert not np.isclose(deleted_edge_gradient, p_a * (1 - p_a))


def test_action_dependent_realized_mask_creates_spurious_gradient():
    p = 0.4
    estimated_gradient = sum(
        (p if action else 1 - p) * (action - p) * action
        for action in (0, 1)
    )
    assert np.isclose(estimated_gradient, p * (1 - p))
    assert estimated_gradient != 0.0


def test_corrupted_potentials_still_conserve_return():
    snapshots = snapshots_from_values(("progress",), ((0,), (1,), (0.25,)))
    for weight in (0.0, 1.0, -3.0, 17.5):
        potential = IdentityPotential(("progress",), np.array([weight]))
        conserved = build_conserved_atoms(snapshots, episode_return=1.0, potential=potential)
        assert conserved.conservation_error < 1e-12
        assert np.isclose(sum(atom.value for atom in conserved.atoms), 1.0)


def test_opaque_route_receives_every_atom():
    conserved, routes = _two_action_fixture(a=1, b=1)
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
