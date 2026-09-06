"""Matched structural and numerical diagnostics; zero slots cannot fake gains."""

from __future__ import annotations

import numpy as np

from recipe.exact.core_exact import compute_exact_credits


def compare_alfworld_credits(conserved, routes, snapshots, potential, placements, data, real_steps):
    graph = compute_exact_credits(conserved, routes, mode="graph")
    temporal = compute_exact_credits(conserved, routes, mode="temporal")
    initial = potential.factor_values(snapshots[0])
    baseline, token_weights = [], []
    counts = {kind: [0.0, 0.0] for kind in ("delta", "closure")}
    span_counts = {kind: [0.0, 0.0] for kind in ("delta", "closure")}
    masses = {kind: [0.0, 0.0] for kind in ("delta", "closure")}
    nonzero_removed = 0.0
    nonzero_total = 0.0
    future_known = dict(zip(potential.factor_ids, initial.tolist(), strict=True))
    for index, (row, start, end) in enumerate(placements):
        schema = data.non_tensor_batch["exact_effect_schema"][row]
        step = int(data.non_tensor_batch["exact_step_id"][row])
        record = next(record for record in schema["spans"] if record["token_start"] == start)
        mutable = set(record["certificate"]["future_factor_ids"])
        values = potential.factor_values(snapshots[step - 1])
        value_map = dict(zip(potential.factor_ids, values.tolist(), strict=True))
        baseline.append(conserved.episode_return - sum(value_map[key] - future_known[key] for key in mutable))
        tokens = end - start
        token_weights.append(tokens)
        graph_ids = set(graph.span_credits[index].descendant_atom_ids)
        temporal_ids = set(temporal.span_credits[index].descendant_atom_ids)
        if not graph_ids <= temporal_ids:
            raise AssertionError("ALFWorld graph is not a subset of its temporal comparison")
        for atom in conserved.atoms:
            if atom.atom_kind == "opaque_target" or atom.atom_id not in temporal_ids:
                continue
            if atom.atom_kind == "delta" and atom.step_id > real_steps:
                continue  # Padding is a proof device, never a sparsity gain.
            counts[atom.atom_kind][1] += tokens
            counts[atom.atom_kind][0] += tokens * (atom.atom_id not in graph_ids)
            span_counts[atom.atom_kind][1] += 1
            span_counts[atom.atom_kind][0] += atom.atom_id not in graph_ids
            masses[atom.atom_kind][1] += tokens * abs(atom.value)
            nonzero_total += tokens * abs(atom.value)
            if atom.atom_id not in graph_ids:
                nonzero_removed += tokens * abs(atom.value)
                masses[atom.atom_kind][0] += tokens * abs(atom.value)
    g = np.asarray([span.credit for span in graph.span_credits])
    t = np.asarray([span.credit for span in temporal.span_credits])
    b = np.asarray(baseline)
    weights = np.asarray(token_weights, dtype=np.float64)
    error = float(np.max(np.abs(g - b)))
    # This implementation certifies constants. Honest equality to the
    # same-information prefix baseline is an invariant, not a novel CV claim.
    if error > 1e-10:
        raise AssertionError("ALFWorld certified graph differs from its analytic prefix baseline")
    metrics = {
        "delta_pair_reduction": counts["delta"][0] / max(counts["delta"][1], 1),
        "closure_pair_reduction": counts["closure"][0] / max(counts["closure"][1], 1),
        "nonzero_mass_removed": nonzero_removed / max(nonzero_total, 1e-30),
        "credit_difference_abs": float(np.average(np.abs(g - t), weights=weights)),
        "credit_difference_nonzero_rate": float(np.average(np.abs(g - t) > 1e-10, weights=weights)),
        "prefix_baseline_error_max": error,
    }
    for kind in counts:
        metrics[f"{kind}_span_pair_reduction"] = span_counts[kind][0] / max(span_counts[kind][1], 1)
        metrics[f"{kind}_pairs_removed"] = counts[kind][0]
        metrics[f"{kind}_pairs_total"] = counts[kind][1]
        metrics[f"{kind}_nonzero_abs_mass_removed"] = masses[kind][0]
        metrics[f"{kind}_nonzero_abs_mass_total"] = masses[kind][1]
    return {"metrics": metrics, "graph_credits": g.tolist(), "temporal_credits": t.tolist(), "prefix_baseline_credits": b.tolist()}
