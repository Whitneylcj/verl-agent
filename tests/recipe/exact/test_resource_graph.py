from recipe.exact.appworld_schema import POLICY_CONTEXT_RESOURCE, UNKNOWN_RESOURCE
from recipe.exact.credit_spec import CreditAtom, EffectSpan, SpanRoute
from recipe.exact.resource_graph import compile_resource_graph_routes


def _route(span_id, step_id, writes, context=(), parents=()):
    return SpanRoute(
        span=EffectSpan(
            span_id=span_id,
            step_id=step_id,
            token_start=0,
            token_end=1,
            possible_write_set=tuple(writes),
            context_sources=tuple(context),
            control_parents=tuple(parents),
        ),
        descendant_atom_ids=(),
        soundness_certificate="test-resource-graph",
        route_kind="resource_graph",
    )


def test_resource_graph_routes_direct_writes_and_propagates_context_control():
    atoms = (
        CreditAtom("venmo-now", 1.0, 1, "venmo", ("model:venmo",)),
        CreditAtom("spotify-now", 1.0, 1, "spotify", ("model:spotify",)),
        CreditAtom("gmail-later", 1.0, 2, "gmail", ("model:gmail",)),
        CreditAtom("unknown-later", 1.0, 2, "unknown", (UNKNOWN_RESOURCE,)),
        CreditAtom("residual", 0.0, None, None, ("episode_return",), True),
    )
    routes = (
        _route("step1-arguments", 1, ("model:venmo", POLICY_CONTEXT_RESOURCE)),
        _route(
            "step2-arguments",
            2,
            ("model:gmail", POLICY_CONTEXT_RESOURCE),
            context=(POLICY_CONTEXT_RESOURCE,),
        ),
    )

    first, second = compile_resource_graph_routes(routes, atoms)
    assert first.route_kind == "explicit"
    assert first.descendant_atom_ids == ("venmo-now", "gmail-later", "unknown-later")
    assert second.descendant_atom_ids == ("gmail-later", "unknown-later")
    assert "spotify-now" not in first.descendant_atom_ids


def test_explicit_control_parent_must_precede_child():
    atoms = (CreditAtom("a", 1.0, 1, "a", ("model:a",)),)
    routes = (
        _route("child", 1, ("model:a",), parents=("parent",)),
        _route("parent", 1, ("model:a",)),
    )
    try:
        compile_resource_graph_routes(routes, atoms)
    except ValueError as error:
        assert "precede" in str(error)
    else:
        raise AssertionError("expected invalid control ordering to fail")
