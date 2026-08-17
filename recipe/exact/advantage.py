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
    build_conserved_atoms,
    compute_exact_credits,
)
from recipe.exact.credit_spec import EffectSpan, FactorSnapshot, SpanRoute


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
            raise ValueError(
                f"snapshot checkpoint {record.checkpoint_id} does not match expected {checkpoint_id}"
            )
        return record
    if not isinstance(record, Mapping):
        raise TypeError("EXACT factor snapshots must be mappings or FactorSnapshot instances")
    recorded_checkpoint = int(record.get("checkpoint_id", checkpoint_id))
    if recorded_checkpoint != checkpoint_id:
        raise ValueError(
            f"snapshot checkpoint {recorded_checkpoint} does not match expected {checkpoint_id}"
        )
    return FactorSnapshot(
        checkpoint_id=checkpoint_id,
        factor_ids=tuple(record["factor_ids"]),
        values=np.asarray(record["values"], dtype=np.float64),
        read_sets={key: tuple(value) for key, value in record.get("read_sets", {}).items()},
        schema_version=str(record.get("schema_version", "exact.credit.v1")),
    )


def _make_potential(factor_ids: Sequence[str], config: Any) -> IdentityPotential:
    potential_config = _config_get(config, "potential", {})
    weights_config = _config_get(potential_config, "weights", None)
    if weights_config is None:
        scale = float(_config_get(potential_config, "scale", 1.0))
        return IdentityPotential.normalized(factor_ids, scale=scale)
    if isinstance(weights_config, Mapping):
        missing = set(factor_ids) - set(weights_config)
        extra = set(weights_config) - set(factor_ids)
        if missing or extra:
            raise ValueError(
                f"potential weight keys must exactly match factor IDs; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        weights = np.asarray([weights_config[factor_id] for factor_id in factor_ids], dtype=np.float64)
    else:
        weights = np.asarray(weights_config, dtype=np.float64)
    return IdentityPotential(tuple(factor_ids), weights)


def _trajectory_snapshots(rows: Sequence[int], data: Any) -> tuple[FactorSnapshot, ...]:
    step_ids = [int(data.non_tensor_batch["exact_step_id"][row]) for row in rows]
    expected = list(range(1, len(rows) + 1))
    if step_ids != expected:
        raise ValueError(f"EXACT step IDs must be contiguous within a trajectory: {step_ids} != {expected}")

    snapshots = [_snapshot_from_record(data.non_tensor_batch["exact_factor_pre"][rows[0]], 0)]
    for position, row in enumerate(rows, start=1):
        recorded_pre = _snapshot_from_record(data.non_tensor_batch["exact_factor_pre"][row], position - 1)
        if recorded_pre.factor_ids != snapshots[-1].factor_ids or not np.allclose(
            recorded_pre.values,
            snapshots[-1].values,
            rtol=0.0,
            atol=1e-10,
        ):
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
            raise ValueError(
                f"invalid effect span [{start}, {end}) for response length {valid_response_length}"
            )
        coverage[start:end] += 1
        opaque = bool(record.get("opaque", "descendant_factor_ids" not in record))
        if opaque:
            descendant_ids = None
            fallback_count += 1
        else:
            factor_ids = tuple(str(value) for value in record["descendant_factor_ids"])
            descendant_ids = tuple(
                atom.atom_id
                for atom in atoms
                if not atom.is_residual
                and atom.step_id is not None
                and atom.step_id >= step_id
                and atom.factor_id in factor_ids
            )
        span = EffectSpan(
            span_id=str(record.get("span_id", f"row:{row}:span:{span_position}")),
            step_id=step_id,
            token_start=start,
            token_end=end,
            bucket=str(record.get("bucket", "default")),
            possible_write_set=tuple(record.get("possible_write_set", ())),
            control_parents=tuple(record.get("control_parents", ())),
            context_sources=tuple(record.get("context_sources", ())),
            opaque=opaque,
        )
        routes.append(
            SpanRoute(
                span=span,
                descendant_atom_ids=descendant_ids,
                soundness_certificate=str(record.get("certificate", "opaque-all-to-all")),
            )
        )
        offsets.append((start, end))

    if valid_response_length <= 0:
        raise ValueError("EXACT cannot score an empty response")
    if not np.all(coverage == 1):
        raise ValueError("effect spans must partition every valid response token exactly once")
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
    mode = str(_config_get(config, "mode", "graph"))
    force_residual = bool(_config_get(config, "force_residual_descendant", True))
    if alpha_by_bucket is None:
        alpha_by_bucket = dict(_config_get(config, "alpha_by_bucket", {}) or {})
    else:
        alpha_by_bucket = dict(alpha_by_bucket)

    conservation_errors: list[float] = []
    residual_ratios: list[float] = []
    cone_densities: list[float] = []
    raw_credits: list[float] = []
    fallback_count = 0
    span_count = 0
    traces: list[dict[str, Any]] = []
    factor_atom_count = 0
    changed_factor_atom_count = 0
    graph_compile_seconds = 0.0

    for trajectory_id, rows in trajectory_rows.items():
        snapshots = _trajectory_snapshots(rows, data)
        potential = _make_potential(snapshots[0].factor_ids, config)
        episode_return = _trajectory_return(rows, data)
        conserved = build_conserved_atoms(
            snapshots,
            episode_return=episode_return,
            potential=potential,
            tolerance=float(_config_get(config, "conservation_tolerance", 1e-8)),
        )
        trajectory_factor_atoms = [atom for atom in conserved.atoms if not atom.is_residual]
        factor_atom_count += len(trajectory_factor_atoms)
        changed_factor_atom_count += sum(abs(atom.value) > 1e-12 for atom in trajectory_factor_atoms)

        graph_started = time.perf_counter()
        routes: list[SpanRoute] = []
        placements: list[tuple[int, int, int]] = []
        for row in rows:
            valid_length = int(base_response_mask[row].sum().item())
            row_routes, offsets, row_fallbacks = _compile_row_routes(
                row=row,
                step_id=int(data.non_tensor_batch["exact_step_id"][row]),
                valid_response_length=valid_length,
                schema=data.non_tensor_batch["exact_effect_schema"][row],
                atoms=conserved.atoms,
            )
            routes.extend(row_routes)
            placements.extend((row, start, end) for start, end in offsets)
            fallback_count += row_fallbacks
            span_count += len(row_routes)

        result = compute_exact_credits(
            conserved,
            routes,
            alpha_by_bucket=alpha_by_bucket,
            mode=mode,
            force_residual_descendant=force_residual,
        )
        graph_compile_seconds += time.perf_counter() - graph_started
        for placement, span_credit in zip(placements, result.span_credits):
            row, start, end = placement
            advantages[row, start:end] = float(span_credit.credit * trajectory_scale)
            raw_credits.append(span_credit.credit)

        conservation_errors.append(conserved.conservation_error)
        residual_ratios.append(conserved.residual_ratio)
        cone_densities.append(result.cone_density)
        traces.append(
            {
                "trajectory_id": trajectory_id,
                "episode_return": episode_return,
                "conservation_error": conserved.conservation_error,
                "residual_ratio": conserved.residual_ratio,
                "cone_density": result.cone_density,
                "atoms": [asdict(atom) for atom in conserved.atoms],
                "spans": [asdict(span_credit) for span_credit in result.span_credits],
            }
        )

    advantages = advantages * exact_response_mask
    data.batch["response_mask"] = exact_response_mask
    data.batch["advantages"] = advantages
    data.batch["returns"] = advantages.clone()
    loss_mask = data.batch.get("loss_mask", data.batch["attention_mask"].clone()).clone()
    loss_mask[:, -response_length:] = exact_response_mask.to(dtype=loss_mask.dtype)
    data.batch["loss_mask"] = loss_mask

    credit_array = np.asarray(raw_credits, dtype=np.float64)
    probe_seconds = 0.0
    if "exact_probe_seconds" in data.non_tensor_batch:
        probe_values = np.asarray(data.non_tensor_batch["exact_probe_seconds"], dtype=np.float64)
        probe_seconds = float(probe_values[~exact_padding].sum())
    metrics = {
        "exact/conservation_error_max": float(max(conservation_errors, default=0.0)),
        "exact/residual_ratio_mean": float(np.mean(residual_ratios)),
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
        "exact/probe_seconds": probe_seconds,
        "exact/graph_compile_seconds": float(graph_compile_seconds),
        "exact/pathwise_conservation_pass": 1.0,
    }
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
