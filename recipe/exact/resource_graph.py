"""Conservative resource/control graph compilation for Exact-G."""

from __future__ import annotations

from typing import Sequence

from recipe.exact.appworld_schema import ALL_RESOURCE, UNKNOWN_RESOURCE
from recipe.exact.credit_spec import CreditAtom, SpanRoute


def resource_sets_overlap(left: Sequence[str], right: Sequence[str]) -> bool:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return False
    if {ALL_RESOURCE, UNKNOWN_RESOURCE} & left_set:
        return True
    if {ALL_RESOURCE, UNKNOWN_RESOURCE} & right_set:
        return True
    return bool(left_set & right_set)


def compile_resource_graph_routes(
    routes: Sequence[SpanRoute],
    atoms: Sequence[CreditAtom],
) -> tuple[SpanRoute, ...]:
    """Resolve resource routes, then propagate descendants over control edges."""

    routes = tuple(routes)
    if not any(route.route_kind == "resource_graph" for route in routes):
        return routes
    atom_ids = tuple(atom.atom_id for atom in atoms)
    atom_id_set = set(atom_ids)
    span_ids = [route.span.span_id for route in routes]
    if len(span_ids) != len(set(span_ids)):
        raise ValueError("resource graph span IDs must be unique within a trajectory")

    descendants: list[set[str]] = []
    for route in routes:
        if route.span.opaque or route.descendant_atom_ids is None:
            descendants.append(set(atom_ids))
            continue
        if route.route_kind == "resource_graph":
            descendants.append({atom.atom_id for atom in atoms if not atom.is_residual and atom.step_id is not None and atom.step_id >= route.span.step_id and resource_sets_overlap(route.span.possible_write_set, atom.read_set)})
            continue
        unknown = set(route.descendant_atom_ids) - atom_id_set
        if unknown:
            raise ValueError(f"route {route.span.span_id} refers to unknown atoms: {sorted(unknown)}")
        descendants.append(set(route.descendant_atom_ids))

    span_position = {span_id: index for index, span_id in enumerate(span_ids)}
    for child_index, route in enumerate(routes):
        for parent_id in route.span.control_parents:
            parent_index = span_position.get(parent_id)
            if parent_index is None:
                raise ValueError(f"route {route.span.span_id} has unknown control parent: {parent_id}")
            if parent_index >= child_index:
                raise ValueError("control parents must precede their child span")

    for parent_index in reversed(range(len(routes))):
        parent = routes[parent_index]
        for child_index in range(parent_index + 1, len(routes)):
            child = routes[child_index]
            explicit_control = parent.span.span_id in child.span.control_parents
            context_control = resource_sets_overlap(
                parent.span.possible_write_set,
                child.span.context_sources,
            )
            if explicit_control or context_control:
                descendants[parent_index].update(descendants[child_index])

    compiled = []
    for route, descendant_ids in zip(routes, descendants, strict=True):
        compiled.append(
            SpanRoute(
                span=route.span,
                descendant_atom_ids=tuple(atom_id for atom_id in atom_ids if atom_id in descendant_ids),
                soundness_certificate=route.soundness_certificate,
                route_kind="explicit",
            )
        )
    return tuple(compiled)
