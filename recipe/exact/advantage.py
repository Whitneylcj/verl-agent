"""Bridge conserved EXACT credits into verl's token-level advantage tensors."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from recipe.exact.core_exact import (
    DelayedAlphaController,
    IdentityPotential,
    build_scoped_conserved_atoms,
    compute_exact_credits,
)
from recipe.exact.credit_spec import EffectSpan, FactorSnapshot, SpanRoute
from recipe.exact.resource_graph import compile_resource_graph_routes


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if getter is not None:
        return getter(key, default)
    return getattr(config, key, default)


def _snapshot_from_record(record: Any, checkpoint_id: int) -> FactorSnapshot:
    if isinstance(record, FactorSnapshot):
        if record.checkpoint_id != checkpoint_id:
            raise ValueError(f"snapshot checkpoint {record.checkpoint_id} does not match expected {checkpoint_id}")
        return record
    if not isinstance(record, Mapping):
        raise TypeError("EXACT factor snapshots must be mappings or FactorSnapshot instances")
    recorded_checkpoint = int(record.get("checkpoint_id", checkpoint_id))
    if recorded_checkpoint != checkpoint_id:
        raise ValueError(f"snapshot checkpoint {recorded_checkpoint} does not match expected {checkpoint_id}")
    return FactorSnapshot(
        checkpoint_id=checkpoint_id,
        factor_ids=tuple(record["factor_ids"]),
        values=np.asarray(record["values"], dtype=np.float64),
        potential_weights=(None if record.get("potential_weights") is None else np.asarray(record["potential_weights"], dtype=np.float64)),
        read_sets={key: tuple(value) for key, value in record.get("read_sets", {}).items()},
        schema_version=str(record.get("schema_version", "exact.credit.v1")),
        channel_roles={key: str(value) for key, value in record["channel_roles"].items()},
        source_revision=(None if record.get("source_revision") is None else str(record["source_revision"])),
    )


def _make_potential(snapshot: FactorSnapshot, config: Any) -> IdentityPotential:
    factor_ids = snapshot.factor_ids
    potential_config = _config_get(config, "potential", {})
    weights_config = _config_get(potential_config, "weights", None)
    scale = float(_config_get(potential_config, "scale", 1.0))
    has_return_components = any(role == "return_component" for role in snapshot.channel_roles.values())
    if has_return_components and scale != 1.0:
        raise ValueError("return_component channels require potential.scale=1.0")
    if weights_config is None:
        if snapshot.potential_weights is not None:
            return IdentityPotential(factor_ids, scale * snapshot.potential_weights)
        return IdentityPotential.normalized(factor_ids, scale=scale)
    if isinstance(weights_config, Mapping):
        missing = set(factor_ids) - set(weights_config)
        extra = set(weights_config) - set(factor_ids)
        if missing or extra:
            raise ValueError(f"potential weight keys must exactly match factor IDs; missing={sorted(missing)}, extra={sorted(extra)}")
        weights = np.asarray([weights_config[factor_id] for factor_id in factor_ids], dtype=np.float64)
    else:
        weights = np.asarray(weights_config, dtype=np.float64)
    potential = IdentityPotential(tuple(factor_ids), weights)
    if has_return_components:
        if snapshot.potential_weights is None or not np.array_equal(weights, snapshot.potential_weights):
            raise ValueError("return_component channels require their exact native weights")
    return potential


def _trajectory_snapshots(rows: Sequence[int], data: Any) -> tuple[FactorSnapshot, ...]:
    step_ids = [int(data.non_tensor_batch["exact_step_id"][row]) for row in rows]
    expected = list(range(1, len(rows) + 1))
    if step_ids != expected:
        raise ValueError(f"EXACT step IDs must be contiguous within a trajectory: {step_ids} != {expected}")

    snapshots = [_snapshot_from_record(data.non_tensor_batch["exact_factor_pre"][rows[0]], 0)]
    for position, row in enumerate(rows, start=1):
        recorded_pre = _snapshot_from_record(data.non_tensor_batch["exact_factor_pre"][row], position - 1)
        previous = snapshots[-1]
        weights_match = (recorded_pre.potential_weights is None) == (previous.potential_weights is None)
        if recorded_pre.potential_weights is not None and previous.potential_weights is not None:
            weights_match = weights_match and np.array_equal(
                recorded_pre.potential_weights,
                previous.potential_weights,
            )
        metadata_match = recorded_pre.factor_ids == previous.factor_ids and recorded_pre.channel_roles == previous.channel_roles and recorded_pre.read_sets == previous.read_sets and recorded_pre.schema_version == previous.schema_version and recorded_pre.source_revision == previous.source_revision
        if not metadata_match or not np.allclose(recorded_pre.values, previous.values, rtol=0.0, atol=1e-10) or not weights_match:
            raise ValueError(f"factor checkpoint discontinuity before trajectory step {position}")
        snapshots.append(_snapshot_from_record(data.non_tensor_batch["exact_factor_post"][row], position))
    return tuple(snapshots)


def _trajectory_return(rows: Sequence[int], data: Any) -> float:
    if "episode_rewards" not in data.non_tensor_batch:
        raise KeyError("EXACT requires the rollout field 'episode_rewards'")
    returns = np.asarray([data.non_tensor_batch["episode_rewards"][row] for row in rows], dtype=np.float64)
    if returns.ndim != 1 or not np.all(np.isfinite(returns)):
        raise ValueError("episode_rewards must be finite scalars")
    if not np.allclose(returns, returns[0], rtol=0.0, atol=1e-10):
        raise ValueError("all rows of an EXACT trajectory must carry the same episode return")
    return float(returns[0])


def _compile_row_routes(
    row: int,
    step_id: int,
    valid_response_length: int,
    schema: Any,
    atoms: Sequence[Any],
) -> tuple[list[SpanRoute], list[tuple[int, int]], int]:
    if schema is None:
        span_records = ({"opaque": True, "certificate": "missing-schema-fallback"},)
    elif isinstance(schema, Mapping) and "spans" in schema:
        span_records = tuple(schema["spans"])
    elif isinstance(schema, Mapping):
        span_records = (schema,)
    else:
        raise TypeError("exact_effect_schema must be a mapping")

    base_span_ids = [str(record.get("span_id", f"span:{span_position}")) for span_position, record in enumerate(span_records)]
    if len(base_span_ids) != len(set(base_span_ids)):
        raise ValueError("effect span IDs must be unique within one response")
    span_id_map = {base_span_id: f"row:{row}:{base_span_id}" for base_span_id in base_span_ids}
    schema_resolution_fallback = bool(isinstance(schema, Mapping) and schema.get("resolution_fallback", False))

    routes: list[SpanRoute] = []
    offsets: list[tuple[int, int]] = []
    coverage = np.zeros(valid_response_length, dtype=np.int64)
    fallback_count = 0
    for span_position, record in enumerate(span_records):
        if not isinstance(record, Mapping):
            raise TypeError("each exact effect span schema must be a mapping")
        start = int(record.get("token_start", 0))
        end = int(record.get("token_end", valid_response_length))
        if start < 0 or end > valid_response_length or end <= start:
            raise ValueError(f"invalid effect span [{start}, {end}) for response length {valid_response_length}")
        coverage[start:end] += 1
        route_kind = str(record.get("route_kind", "explicit"))
        opaque = bool(
            record.get(
                "opaque",
                route_kind != "resource_graph" and "descendant_factor_ids" not in record,
            )
        )
        if opaque:
            descendant_ids = None
            fallback_count += 1
        elif route_kind == "resource_graph":
            descendant_ids = ()
        else:
            factor_ids = tuple(str(value) for value in record["descendant_factor_ids"])
            descendant_ids = tuple(atom.atom_id for atom in atoms if atom.atom_kind in {"delta", "closure"} and atom.step_id >= step_id and atom.channel_id in factor_ids)
        span = EffectSpan(
            span_id=span_id_map[base_span_ids[span_position]],
            step_id=step_id,
            token_start=start,
            token_end=end,
            bucket=str(record.get("bucket", "default")),
            possible_write_set=tuple(record.get("possible_write_set", ())),
            control_parents=tuple(span_id_map.get(str(parent), str(parent)) for parent in record.get("control_parents", ())),
            context_sources=tuple(record.get("context_sources", ())),
            opaque=opaque,
        )
        routes.append(
            SpanRoute(
                span=span,
                descendant_atom_ids=descendant_ids,
                soundness_certificate=str(record.get("certificate", "opaque-all-to-all")),
                route_kind=route_kind,
            )
        )
        offsets.append((start, end))

    if valid_response_length <= 0:
        raise ValueError("EXACT cannot score an empty response")
    if not np.all(coverage == 1):
        raise ValueError("effect spans must partition every valid response token exactly once")
    if schema_resolution_fallback and fallback_count == 0:
        fallback_count += 1
    return routes, offsets, fallback_count


def compute_exact_advantage(
    data: Any,
    config: Any = None,
    alpha_by_bucket: Mapping[str, float] | None = None,
) -> tuple[Any, dict[str, float], list[dict[str, Any]]]:
    """Compile trajectory credit metadata into token-level advantages.

    The actor uses ``seq-mean-token-sum``. Each credit is therefore multiplied
    by ``padded_rows / unique_trajectories`` so the batch mean equals a mean of
    per-trajectory sums. Rows added only for data-parallel divisibility receive
    a zero loss mask and never enter credit construction.
    """

    required = {
        "traj_uid",
        "exact_step_id",
        "exact_factor_pre",
        "exact_factor_post",
        "exact_effect_schema",
        "episode_rewards",
    }
    missing = required - set(data.non_tensor_batch)
    if missing:
        raise KeyError(f"EXACT rollout metadata is missing: {sorted(missing)}")

    batch_size, response_length = data.batch["responses"].shape
    base_response_mask = data.batch.get("response_mask")
    if base_response_mask is None:
        base_response_mask = data.batch["attention_mask"][:, -response_length:]
    exact_padding = np.asarray(
        data.non_tensor_batch.get("exact_padding", np.zeros(batch_size, dtype=bool)),
        dtype=bool,
    )
    if exact_padding.shape != (batch_size,):
        raise ValueError("exact_padding must have shape [batch]")

    trajectory_rows: dict[str, list[int]] = {}
    for row in range(batch_size):
        if exact_padding[row]:
            continue
        trajectory_rows.setdefault(str(data.non_tensor_batch["traj_uid"][row]), []).append(row)
    if not trajectory_rows:
        raise ValueError("EXACT batch contains no non-padding trajectories")
    for rows in trajectory_rows.values():
        rows.sort(key=lambda row: int(data.non_tensor_batch["exact_step_id"][row]))

    advantages = torch.zeros_like(base_response_mask, dtype=torch.float32)
    exact_response_mask = base_response_mask.clone().to(dtype=torch.float32)
    exact_response_mask[torch.as_tensor(exact_padding, device=exact_response_mask.device)] = 0
    trajectory_scale = batch_size / len(trajectory_rows)
    conservation_schema = str(_config_get(config, "conservation_schema", "scoped_v2"))
    if conservation_schema != "scoped_v2":
        raise ValueError(f"unsupported EXACT conservation schema: {conservation_schema}")
    mode = str(_config_get(config, "mode", "graph"))
    if alpha_by_bucket is None:
        alpha_by_bucket = dict(_config_get(config, "alpha_by_bucket", {}) or {})
    else:
        alpha_by_bucket = dict(alpha_by_bucket)

    conservation_errors: list[float] = []
    closure_abs_masses: list[float] = []
    closure_abs_ratios: list[float] = []
    opaque_target_abs_ratios: list[float] = []
    cone_densities: list[float] = []
    closure_route_densities: list[float] = []
    raw_credits: list[float] = []
    fallback_count = 0
    span_count = 0
    traces: list[dict[str, Any]] = []
    factor_atom_count = 0
    changed_factor_atom_count = 0
    graph_compile_seconds = 0.0
    resource_graph_span_count = 0
    unknown_read_atom_count = 0
    appworld_row_count = 0
    appworld_argument_span_count = 0
    appworld_opaque_factor_rates: list[float] = []
    appworld_version_supported: list[float] = []
    appworld_source_supported: list[float] = []
    appworld_factor_compile_fallbacks: list[float] = []

    for trajectory_id, rows in trajectory_rows.items():
        snapshots = _trajectory_snapshots(rows, data)
        potential = _make_potential(snapshots[0], config)
        episode_return = _trajectory_return(rows, data)
        conserved = build_scoped_conserved_atoms(
            snapshots,
            episode_return=episode_return,
            potential=potential,
            tolerance=float(_config_get(config, "conservation_tolerance", 1e-8)),
        )
        if snapshots[0].schema_version == "exact.sokoban.official_reward.v1":
            tolerance = float(_config_get(config, "conservation_tolerance", 1e-8))
            if conserved.closure_abs_mass > tolerance:
                raise AssertionError("Sokoban official return-component closures must be zero")
            if abs(conserved.opaque_target_atom.value) > tolerance:
                raise AssertionError("Sokoban official reward events do not reconstruct the episode return")
        trajectory_factor_atoms = [atom for atom in conserved.atoms if atom.atom_kind == "delta"]
        factor_atom_count += len(trajectory_factor_atoms)
        changed_factor_atom_count += sum(abs(atom.value) > 1e-12 for atom in trajectory_factor_atoms)
        unknown_read_atom_count += sum("exact.resource.unknown" in atom.read_set for atom in trajectory_factor_atoms)

        graph_started = time.perf_counter()
        routes: list[SpanRoute] = []
        placements: list[tuple[int, int, int]] = []
        for row in rows:
            row_schema = data.non_tensor_batch["exact_effect_schema"][row]
            if isinstance(row_schema, Mapping) and row_schema.get("kind") == "appworld-concrete-effect-v1":
                appworld_row_count += 1
                factor_count = int(row_schema.get("factor_count", 0))
                opaque_factor_count = int(row_schema.get("opaque_factor_count", 0))
                appworld_opaque_factor_rates.append(opaque_factor_count / max(factor_count, 1))
                appworld_version_supported.append(float(bool(row_schema.get("version_supported", False))))
                appworld_source_supported.append(float(bool(row_schema.get("source_supported", False))))
                appworld_factor_compile_fallbacks.append(float(bool(row_schema.get("factor_schema_compile_fallback", False))))
            valid_length = int(base_response_mask[row].sum().item())
            row_routes, offsets, row_fallbacks = _compile_row_routes(
                row=row,
                step_id=int(data.non_tensor_batch["exact_step_id"][row]),
                valid_response_length=valid_length,
                schema=data.non_tensor_batch["exact_effect_schema"][row],
                atoms=conserved.atoms,
            )
            routes.extend(row_routes)
            resource_graph_span_count += sum(route.route_kind == "resource_graph" for route in row_routes)
            appworld_argument_span_count += sum(route.span.bucket == "appworld.arguments" for route in row_routes)
            placements.extend((row, start, end) for start, end in offsets)
            fallback_count += row_fallbacks
            span_count += len(row_routes)

        routes = list(compile_resource_graph_routes(routes, conserved.atoms))
        result = compute_exact_credits(
            conserved,
            routes,
            alpha_by_bucket=alpha_by_bucket,
            mode=mode,
        )
        graph_compile_seconds += time.perf_counter() - graph_started
        for placement, span_credit in zip(placements, result.span_credits):
            row, start, end = placement
            advantages[row, start:end] = float(span_credit.credit * trajectory_scale)
            raw_credits.append(span_credit.credit)

        conservation_errors.append(conserved.conservation_error)
        closure_abs_masses.append(conserved.closure_abs_mass)
        closure_abs_ratios.append(conserved.closure_abs_ratio)
        opaque_target_abs_ratios.append(conserved.opaque_target_abs_ratio)
        cone_densities.append(result.cone_density)
        closure_route_densities.append(result.closure_route_density)
        channel_identities = {}
        for channel_id, target_mass in conserved.channel_target_masses.items():
            delta_sum = sum(atom.value for atom in conserved.atoms if atom.atom_kind == "delta" and atom.channel_id == channel_id)
            closure_value = next(atom.value for atom in conserved.closure_atoms if atom.channel_id == channel_id)
            channel_identities[channel_id] = {
                "delta_sum": float(delta_sum),
                "closure": float(closure_value),
                "target_mass": float(target_mass),
                "conservation_error": conserved.channel_conservation_errors[channel_id],
            }
        traces.append(
            {
                "trajectory_id": trajectory_id,
                "episode_return": episode_return,
                "atom_sum": float(sum(atom.value for atom in conserved.atoms)),
                "conservation_error": conserved.conservation_error,
                "channel_roles": dict(snapshots[0].channel_roles),
                "channel_target_masses": dict(conserved.channel_target_masses),
                "channel_conservation_errors": dict(conserved.channel_conservation_errors),
                "channel_identities": channel_identities,
                "closure_abs_mass": conserved.closure_abs_mass,
                "closure_abs_ratio": conserved.closure_abs_ratio,
                "opaque_target_abs_ratio": conserved.opaque_target_abs_ratio,
                "cone_density": result.cone_density,
                "closure_route_density": result.closure_route_density,
                "atoms": [asdict(atom) for atom in conserved.atoms],
                "spans": [asdict(span_credit) for span_credit in result.span_credits],
            }
        )

    advantages = advantages * exact_response_mask
    data.batch["response_mask"] = exact_response_mask
    data.batch["advantages"] = advantages
    data.batch["returns"] = advantages.clone()
    data.batch["exact_aux_loss_scale"] = torch.full(
        (batch_size,),
        float(trajectory_scale),
        dtype=torch.float32,
        device=advantages.device,
    )
    loss_mask = data.batch.get("loss_mask", data.batch["attention_mask"].clone()).clone()
    loss_mask[:, -response_length:] = exact_response_mask.to(dtype=loss_mask.dtype)
    data.batch["loss_mask"] = loss_mask

    credit_array = np.asarray(raw_credits, dtype=np.float64)
    probe_seconds = 0.0
    if "exact_probe_seconds" in data.non_tensor_batch:
        probe_values = np.asarray(data.non_tensor_batch["exact_probe_seconds"], dtype=np.float64)
        probe_seconds = float(probe_values[~exact_padding].sum())
    probe_count = 0.0
    if "exact_probe_count" in data.non_tensor_batch:
        probe_counts = np.asarray(data.non_tensor_batch["exact_probe_count"], dtype=np.float64)
        probe_count = float(probe_counts[~exact_padding].sum())
    metrics = {
        "exact/conservation_error_max": float(max(conservation_errors, default=0.0)),
        "exact/closure_abs_mass_mean": float(np.mean(closure_abs_masses)),
        "exact/closure_abs_ratio_mean": float(np.mean(closure_abs_ratios)),
        "exact/opaque_target_abs_ratio_mean": float(np.mean(opaque_target_abs_ratios)),
        "exact/closure_route_density_mean": float(np.mean(closure_route_densities)),
        "exact/cone_density_mean": float(np.mean(cone_densities)),
        "exact/schema_fallback_rate": float(fallback_count / max(span_count, 1)),
        "exact/credit_mean": float(np.mean(credit_array)),
        "exact/credit_std": float(np.std(credit_array)),
        "exact/credit_min": float(np.min(credit_array)),
        "exact/credit_max": float(np.max(credit_array)),
        "exact/trajectory_count": float(len(trajectory_rows)),
        "exact/env_step_count": float(sum(len(rows) for rows in trajectory_rows.values())),
        "exact/generated_token_count": float(exact_response_mask.sum().item()),
        "exact/padding_rows": float(exact_padding.sum()),
        "exact/trajectory_scale": float(trajectory_scale),
        "exact/factor_change_rate": float(changed_factor_atom_count / max(factor_atom_count, 1)),
        "exact/resource_graph_span_rate": float(resource_graph_span_count / max(span_count, 1)),
        "exact/unknown_factor_read_rate": float(unknown_read_atom_count / max(factor_atom_count, 1)),
        "exact/probe_seconds": probe_seconds,
        "exact/verifier_snapshot_count": probe_count,
        "exact/probe_seconds_per_snapshot": probe_seconds / probe_count if probe_count > 0 else 0.0,
        "exact/graph_compile_seconds": float(graph_compile_seconds),
        "exact/pathwise_conservation_pass": 1.0,
        "exact/conservation_schema_scoped_v2": 1.0,
    }
    if appworld_row_count:
        metrics.update(
            {
                "exact/appworld_argument_span_rate": float(appworld_argument_span_count / appworld_row_count),
                "exact/appworld_factor_opaque_rate": float(np.mean(appworld_opaque_factor_rates)),
                "exact/appworld_version_supported": float(min(appworld_version_supported)),
                "exact/appworld_source_supported": float(min(appworld_source_supported)),
                "exact/appworld_factor_compile_fallback_rate": float(np.mean(appworld_factor_compile_fallbacks)),
            }
        )
    for quantile in (5, 25, 50, 75, 95):
        metrics[f"exact/credit_p{quantile:02d}"] = float(np.percentile(credit_array, quantile))
    for bucket, alpha in alpha_by_bucket.items():
        metrics[f"exact/cv_alpha_active/{bucket}"] = float(alpha)
    return data, metrics, traces


def fit_delayed_alpha_from_traces(
    controller: DelayedAlphaController,
    traces: Sequence[Mapping[str, Any]],
    min_samples: int = 8,
) -> Mapping[str, float] | None:
    """Fit next-batch CV coefficients from detached scalar credit proxies.

    This uses causal/control returns as a low-cost proxy for full parameter
    gradient covariance. Delaying activation keeps the estimator unbiased; the
    proxy affects variance only and is explicitly reported in configuration.
    """

    spans = [span for trace in traces for span in trace["spans"]]
    if len(spans) < min_samples:
        return None
    bucket_position = {bucket: index for index, bucket in enumerate(controller.bucket_names)}
    hard = np.asarray([[float(span["causal_return"])] for span in spans], dtype=np.float64)
    controls = np.zeros((len(spans), len(controller.bucket_names), 1), dtype=np.float64)
    for sample, span in enumerate(spans):
        bucket = str(span["bucket"])
        if bucket not in bucket_position:
            raise ValueError(f"credit trace uses an unconfigured CV bucket: {bucket}")
        controls[sample, bucket_position[bucket], 0] = float(span["control_return"])
    return controller.fit_pending(hard, controls)
