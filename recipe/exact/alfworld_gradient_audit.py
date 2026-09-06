"""Measure matched raw-score gradient variance; never substitute credit variance.

Input NPZ contains scores [episodes, tokens, coordinates] computed at the
sampling policy, mask [episodes, tokens], graph/temporal/prefix_baseline credits
[episodes, tokens], and returns [episodes]. Coordinates must be full parameters
or a fixed, data-independent projection declared through --coordinates.
Do not supply gradients after PPO clipping, KL, or optimizer updates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def audit_gradient_samples(scores, mask, credits, returns):
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(mask)
    returns = np.asarray(returns, dtype=np.float64)
    if scores.ndim != 3 or scores.shape[0] < 2 or mask.shape != scores.shape[:2] or returns.shape != scores.shape[:1]:
        raise ValueError("need >= 2 trajectories with aligned score/mask/return arrays")
    if not np.isfinite(scores).all() or not np.isfinite(returns).all() or not np.isin(mask, [0, 1]).all():
        raise ValueError("scores/returns must be finite and masks binary")
    if set(credits) != {"graph", "temporal", "prefix_baseline"}:
        raise ValueError("matched graph, temporal and prefix_baseline credits are required")
    credit_arrays = {key: np.asarray(value, dtype=np.float64) for key, value in credits.items()}
    if any(value.shape != mask.shape or not np.isfinite(value).all() for value in credit_arrays.values()):
        raise ValueError("credits must be finite and aligned with valid score tokens")
    credit_arrays["official"] = np.broadcast_to(returns[:, None], mask.shape)
    gradients = {key: np.einsum("ntp,nt->np", scores, value * mask) for key, value in credit_arrays.items()}
    result = {key: {"variance_trace": float(np.var(value, axis=0, ddof=1).sum()), "mean_gradient": value.mean(axis=0).tolist()} for key, value in gradients.items()}
    for key in ("graph", "temporal", "prefix_baseline"):
        difference = gradients[key] - gradients["official"]
        result[key]["paired_mean_difference_norm"] = float(np.linalg.norm(difference.mean(axis=0)))
        result[key]["paired_standard_error_norm"] = float(np.sqrt(np.var(difference, axis=0, ddof=1).sum() / len(difference)))
    return {"trajectory_count": len(returns), "coordinate_count": scores.shape[-1], "estimators": result, "graph_prefix_sample_difference_max": float(np.max(np.abs(gradients["graph"] - gradients["prefix_baseline"])))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coordinates", choices=("full_parameters", "fixed_projection"), required=True)
    parser.add_argument("--sampling-checkpoint", required=True)
    parser.add_argument("--harness-id", required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as arrays:
        result = audit_gradient_samples(arrays["scores"], arrays["mask"], {key: arrays[key] for key in ("graph", "temporal", "prefix_baseline")}, arrays["returns"])
    result.update(coordinates=args.coordinates, sampling_checkpoint=args.sampling_checkpoint, harness_id=args.harness_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
