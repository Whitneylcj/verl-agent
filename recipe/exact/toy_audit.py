"""Enumerate the paper's stage-0 Toy SCM and issue a training go/no-go report."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from recipe.exact.core_exact import (
    IdentityPotential,
    build_scoped_conserved_atoms,
    compute_exact_credits,
)
from recipe.exact.credit_spec import EffectSpan, SpanRoute, snapshots_from_values

FACTOR_IDS = ("x", "y", "goal", "coin")
READ_SETS = {
    "x": ("state:x",),
    "y": ("state:y",),
    "goal": ("state:x", "state:y", "control:selector"),
    "coin": ("environment:coin",),
}
BASE_ACTION_NAMES = ("write_x", "overwrite_x", "write_y", "select_goal")


@dataclass(frozen=True)
class ToyTrajectory:
    probability: float
    scores: np.ndarray
    episode_return: float
    conserved: Any
    routes: tuple[SpanRoute, ...]


def _policy_probabilities(distractor_count: int) -> tuple[float, ...]:
    if distractor_count < 0:
        raise ValueError("distractor_count must be non-negative")
    return (0.25, 0.5, 0.5, 0.5) + (0.5,) * distractor_count


def _state_probability(bits: Sequence[int], probabilities: Sequence[float]) -> float:
    return float(math.prod(probability if bit else 1.0 - probability for bit, probability in zip(bits, probabilities)))


def _simulate_snapshots(actions: Sequence[int], coin: int, potential_weights: Sequence[float]):
    x = y = goal = coin_value = 0
    values = [(x, y, goal, coin_value)]
    final_step = len(actions) - 1
    for step, action in enumerate(actions):
        if step == 0:
            x = action
        elif step == 1 and action:
            x = 1
        elif step == 2:
            y = action
        elif step == 3:
            goal = y if action else x
        if step == final_step:
            coin_value = coin
        values.append((x, y, goal, coin_value))
    return snapshots_from_values(
        FACTOR_IDS,
        values,
        read_sets=READ_SETS,
        channel_roles={factor_id: "return_component" for factor_id in FACTOR_IDS},
        potential_weights=potential_weights,
    )


def _sound_routes(action_count: int, atoms: Sequence[Any]) -> tuple[SpanRoute, ...]:
    factor_descendants = (
        frozenset(("x", "goal")),
        frozenset(("x", "goal")),
        frozenset(("y", "goal")),
        frozenset(("goal",)),
    )
    routes = []
    for action_index in range(action_count):
        step_id = action_index + 1
        factors = factor_descendants[action_index] if action_index < len(factor_descendants) else frozenset()
        descendants = [atom.atom_id for atom in atoms if atom.atom_kind in {"delta", "closure"} and atom.channel_id in factors and atom.step_id >= step_id]
        action_name = BASE_ACTION_NAMES[action_index] if action_index < len(BASE_ACTION_NAMES) else f"distractor_{action_index - len(BASE_ACTION_NAMES)}"
        routes.append(
            SpanRoute(
                span=EffectSpan(
                    span_id=f"span:{action_name}",
                    step_id=step_id,
                    token_start=action_index,
                    token_end=action_index + 1,
                    bucket=action_name,
                ),
                descendant_atom_ids=tuple(descendants),
                soundness_certificate="enumerated-static-possible-effects",
            )
        )
    return tuple(routes)


def _enumerate_trajectories(
    distractor_count: int,
    potential_weights: Sequence[float],
    coin_reward: float = 0.25,
) -> tuple[ToyTrajectory, ...]:
    probabilities = _policy_probabilities(distractor_count)
    potential = IdentityPotential(FACTOR_IDS, np.asarray(potential_weights, dtype=np.float64))
    trajectories = []
    for actions in itertools.product((0, 1), repeat=len(probabilities)):
        action_probability = _state_probability(actions, probabilities)
        scores = np.asarray(actions, dtype=np.float64) - np.asarray(probabilities, dtype=np.float64)
        for coin in (0, 1):
            snapshots = _simulate_snapshots(actions, coin, potential_weights)
            episode_return = float(snapshots[-1].values[2] + coin_reward * coin)
            conserved = build_scoped_conserved_atoms(
                snapshots,
                episode_return,
                potential,
                tolerance=1e-12,
            )
            trajectories.append(
                ToyTrajectory(
                    probability=action_probability * 0.5,
                    scores=scores,
                    episode_return=episode_return,
                    conserved=conserved,
                    routes=_sound_routes(len(actions), conserved.atoms),
                )
            )
    probability_sum = sum(trajectory.probability for trajectory in trajectories)
    if not np.isclose(probability_sum, 1.0, atol=1e-12):
        raise AssertionError(f"enumerated probability mass is {probability_sum}")
    return tuple(trajectories)


def _mutate_routes(
    routes: Sequence[SpanRoute],
    atoms: Sequence[Any],
    add_fraction: float = 0.0,
    delete_fraction: float = 0.0,
    seed: int = 2718,
) -> tuple[SpanRoute, ...]:
    if not 0.0 <= add_fraction <= 1.0 or not 0.0 <= delete_fraction <= 1.0:
        raise ValueError("edge corruption fractions must lie in [0, 1]")
    all_ids = tuple(atom.atom_id for atom in atoms)
    true_pairs = [(route_index, atom_id) for route_index, route in enumerate(routes) for atom_id in route.descendant_atom_ids]
    false_pairs = [(route_index, atom_id) for route_index, route in enumerate(routes) for atom_id in all_ids if atom_id not in route.descendant_atom_ids]
    rng = np.random.default_rng(seed)
    true_pairs = [true_pairs[index] for index in rng.permutation(len(true_pairs))]
    false_pairs = [false_pairs[index] for index in rng.permutation(len(false_pairs))]
    remove = set(true_pairs[: round(delete_fraction * len(true_pairs))])
    add = set(false_pairs[: round(add_fraction * len(false_pairs))])
    mutated = []
    for route_index, route in enumerate(routes):
        descendants = [atom_id for atom_id in all_ids if ((route_index, atom_id) in add or atom_id in route.descendant_atom_ids) and (route_index, atom_id) not in remove]
        mutated.append(
            SpanRoute(
                span=route.span,
                descendant_atom_ids=tuple(descendants),
                soundness_certificate=f"toy-add-{add_fraction:.2f}-delete-{delete_fraction:.2f}",
            )
        )
    return tuple(mutated)


def _credit_vector(
    trajectory: ToyTrajectory,
    estimator: str,
    route_transform: Callable[[ToyTrajectory], Sequence[SpanRoute]] | None = None,
) -> np.ndarray:
    if estimator == "outcome":
        return np.full_like(trajectory.scores, trajectory.episode_return)
    routes = trajectory.routes if route_transform is None else tuple(route_transform(trajectory))
    mode = "temporal" if estimator == "exact_t" else "graph"
    result = compute_exact_credits(
        trajectory.conserved,
        routes,
        mode=mode,
    )
    return np.asarray([span.credit for span in result.span_credits], dtype=np.float64)


def _gradient_samples(
    trajectories: Sequence[ToyTrajectory],
    estimator: str,
    route_transform: Callable[[ToyTrajectory], Sequence[SpanRoute]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.asarray([trajectory.probability for trajectory in trajectories], dtype=np.float64)
    credits = np.stack([_credit_vector(trajectory, estimator, route_transform) for trajectory in trajectories])
    scores = np.stack([trajectory.scores for trajectory in trajectories])
    return probabilities, scores * credits, credits


def _population_stats(
    trajectories: Sequence[ToyTrajectory],
    true_gradient: np.ndarray,
    estimator: str,
    route_transform: Callable[[ToyTrajectory], Sequence[SpanRoute]] | None = None,
) -> dict[str, Any]:
    probabilities, samples, credits = _gradient_samples(trajectories, estimator, route_transform)
    mean = np.einsum("s,sd->d", probabilities, samples)
    centered = samples - mean
    trace_variance = float(np.einsum("s,sd,sd->", probabilities, centered, centered))
    bias = mean - true_gradient
    mse = float(np.einsum("s,sd,sd->", probabilities, samples - true_gradient, samples - true_gradient))
    signal = float(np.linalg.norm(true_gradient))
    distractor_credit_max = float(np.max(np.abs(credits[:, len(BASE_ACTION_NAMES) :]))) if credits.shape[1] > 4 else 0.0
    return {
        "mean_gradient": mean.tolist(),
        "bias": bias.tolist(),
        "max_abs_bias": float(np.max(np.abs(bias))),
        "trace_variance": trace_variance,
        "mse": mse,
        "snr": signal / math.sqrt(trace_variance) if trace_variance > 0 else None,
        "distractor_credit_max": distractor_credit_max,
    }


def _monte_carlo_ci(
    trajectories: Sequence[ToyTrajectory],
    true_gradient: np.ndarray,
    estimator: str,
    seed: int,
    sample_count: int,
    route_transform: Callable[[ToyTrajectory], Sequence[SpanRoute]] | None = None,
) -> dict[str, Any]:
    probabilities, population, _ = _gradient_samples(trajectories, estimator, route_transform)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(trajectories), size=sample_count, p=probabilities)
    samples = population[indices]
    mean = samples.mean(axis=0)
    standard_error = samples.std(axis=0, ddof=1) / math.sqrt(sample_count)
    critical = NormalDist().inv_cdf(1.0 - 0.05 / (2.0 * len(true_gradient)))
    lower = mean - critical * standard_error
    upper = mean + critical * standard_error
    covered = np.logical_and(true_gradient >= lower - 1e-15, true_gradient <= upper + 1e-15)
    return {
        "sample_count": sample_count,
        "seed": seed,
        "interval": "Bonferroni simultaneous 95% normal CI",
        "mean_gradient": mean.tolist(),
        "standard_error": standard_error.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "all_coordinates_cover_truth": bool(np.all(covered)),
    }


def _git_state() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_status = subprocess.run(
        [
            "git",
            "status",
            "--short",
            "--untracked-files=all",
            "--",
            "recipe/exact",
            "verl",
            "agent_system",
            "examples/exact_trainer",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": commit, "tracked_clean": not tracked_status, "tracked_status": tracked_status.splitlines()}


def run_toy_audit(
    *,
    base_distractors: int = 2,
    max_distractors: int = 6,
    monte_carlo_samples: int = 100_000,
    seed: int = 17,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Return the complete, deterministic stage-0 audit report."""

    if base_distractors < 0:
        raise ValueError("base_distractors must be non-negative")
    if max_distractors < max(base_distractors, 1):
        raise ValueError("max_distractors must cover base_distractors and at least one scaling point")
    if monte_carlo_samples < 2:
        raise ValueError("monte_carlo_samples must be at least two")
    coin_reward = 0.25
    exact_weights = (0.0, 0.0, 1.0, coin_reward)
    trajectories = _enumerate_trajectories(base_distractors, exact_weights, coin_reward)
    probabilities = np.asarray([trajectory.probability for trajectory in trajectories])
    outcome_samples = np.stack([trajectory.scores * trajectory.episode_return for trajectory in trajectories])
    true_gradient = np.einsum("s,sd->d", probabilities, outcome_samples)

    sound_estimators = {estimator: _population_stats(trajectories, true_gradient, estimator) for estimator in ("outcome", "exact_t", "exact_g")}
    sound_estimators["conservative_supergraph"] = _population_stats(
        trajectories,
        true_gradient,
        "exact_g",
        route_transform=lambda trajectory: _mutate_routes(trajectory.routes, trajectory.conserved.atoms, add_fraction=1.0),
    )
    monte_carlo = {estimator: _monte_carlo_ci(trajectories, true_gradient, estimator, seed, monte_carlo_samples) for estimator in ("outcome", "exact_t", "exact_g")}
    monte_carlo["conservative_supergraph"] = _monte_carlo_ci(
        trajectories,
        true_gradient,
        "exact_g",
        seed,
        monte_carlo_samples,
        route_transform=lambda trajectory: _mutate_routes(
            trajectory.routes,
            trajectory.conserved.atoms,
            add_fraction=1.0,
        ),
    )

    corruption_fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
    false_edge_curve = []
    deleted_edge_curve = []
    for fraction in corruption_fractions:
        false_edge_curve.append(
            {
                "fraction": fraction,
                **_population_stats(
                    trajectories,
                    true_gradient,
                    "exact_g",
                    route_transform=lambda trajectory, f=fraction: _mutate_routes(trajectory.routes, trajectory.conserved.atoms, add_fraction=f),
                ),
            }
        )
        deleted_edge_curve.append(
            {
                "fraction": fraction,
                **_population_stats(
                    trajectories,
                    true_gradient,
                    "exact_g",
                    route_transform=lambda trajectory, f=fraction: _mutate_routes(trajectory.routes, trajectory.conserved.atoms, delete_fraction=f),
                ),
            }
        )

    rng = np.random.default_rng(seed)
    potential_weights = {
        "exact": exact_weights,
        "noisy": (0.15, -0.1, 0.8, 0.4),
        "random": tuple(rng.normal(size=len(FACTOR_IDS)).tolist()),
        "sign_flipped": tuple(-value for value in exact_weights),
        "zero": (0.0,) * len(FACTOR_IDS),
    }
    potential_results = {}
    maximum_conservation_error = 0.0
    for name, weights in potential_weights.items():
        variant = _enumerate_trajectories(base_distractors, weights, coin_reward)
        maximum_conservation_error = max(
            maximum_conservation_error,
            max(trajectory.conserved.conservation_error for trajectory in variant),
        )
        potential_results[name] = {
            "weights": list(weights),
            **_population_stats(variant, true_gradient, "exact_g"),
        }

    scaling_counts = sorted({count for count in (0, 1, 2, 4, max_distractors) if count <= max_distractors})
    scaling = []
    for distractor_count in scaling_counts:
        scaled_trajectories = _enumerate_trajectories(distractor_count, exact_weights, coin_reward)
        scaled_probabilities = np.asarray([trajectory.probability for trajectory in scaled_trajectories])
        scaled_outcome = np.stack([trajectory.scores * trajectory.episode_return for trajectory in scaled_trajectories])
        scaled_truth = np.einsum("s,sd->d", scaled_probabilities, scaled_outcome)
        scaling.append(
            {
                "distractor_count": distractor_count,
                "temporal_horizon": len(BASE_ACTION_NAMES) + distractor_count,
                "causal_action_count": len(BASE_ACTION_NAMES),
                "outcome": _population_stats(scaled_trajectories, scaled_truth, "outcome"),
                "exact_g": _population_stats(scaled_trajectories, scaled_truth, "exact_g"),
            }
        )

    dynamic_probability = 0.4
    dynamic_mask_gradient = sum((dynamic_probability if action else 1.0 - dynamic_probability) * (action - dynamic_probability) * action for action in (0, 1))
    dynamic_expected = dynamic_probability * (1.0 - dynamic_probability)

    outcome_variances = [point["outcome"]["trace_variance"] for point in scaling]
    exact_variances = [point["exact_g"]["trace_variance"] for point in scaling]
    potential_variances = [result["trace_variance"] for result in potential_results.values()]
    checks = {
        "pathwise_conservation": maximum_conservation_error < tolerance,
        "sound_gradient_exactness": all(result["max_abs_bias"] < tolerance for result in sound_estimators.values()),
        "monte_carlo_ci": all(result["all_coordinates_cover_truth"] for result in monte_carlo.values()),
        "false_edges_are_safe": all(result["max_abs_bias"] < tolerance for result in false_edge_curve),
        "deleted_edges_are_biased": deleted_edge_curve[-1]["max_abs_bias"] > 1e-3 and np.allclose(deleted_edge_curve[-1]["bias"], -true_gradient, atol=tolerance),
        "dynamic_mask_counterexample": abs(dynamic_mask_gradient - dynamic_expected) < tolerance and abs(dynamic_mask_gradient) > tolerance,
        "potential_corruption_preserves_mean": all(result["max_abs_bias"] < tolerance for result in potential_results.values()),
        "potential_corruption_changes_variance": max(potential_variances) - min(potential_variances) > 1e-6,
        "distractor_credit_is_zero": all(point["exact_g"]["distractor_credit_max"] < tolerance for point in scaling),
        "temporal_distractors_raise_outcome_variance": all(after > before + tolerance for before, after in zip(outcome_variances, outcome_variances[1:])),
        "exact_g_variance_ignores_distractors": max(exact_variances) - min(exact_variances) < tolerance,
    }
    return {
        "schema_version": "exact-toy-audit/v1",
        "status": "pass" if all(checks.values()) else "no_go",
        "git": _git_state(),
        "environment": {
            "description": "overwritable x/y modules, selector control branch, irrelevant actions, terminal coin flip",
            "factor_ids": list(FACTOR_IDS),
            "base_action_names": list(BASE_ACTION_NAMES),
            "base_distractors": base_distractors,
            "coin_probability": 0.5,
            "coin_reward": coin_reward,
            "trajectory_count": len(trajectories),
        },
        "true_gradient": true_gradient.tolist(),
        "checks": checks,
        "maximum_conservation_error": maximum_conservation_error,
        "sound_estimators": sound_estimators,
        "monte_carlo": monte_carlo,
        "graph_corruption": {
            "seed": 2718,
            "false_edge_curve": false_edge_curve,
            "deleted_edge_curve": deleted_edge_curve,
        },
        "dynamic_mask_counterexample": {
            "probability": dynamic_probability,
            "estimated_gradient": dynamic_mask_gradient,
            "expected_spurious_gradient": dynamic_expected,
        },
        "potential_corruption": potential_results,
        "temporal_distractor_scaling": scaling,
    }


def _write_json_atomic(path: Path, report: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_toy_audit(path: str | Path) -> dict[str, Any]:
    audit_path = Path(path).expanduser().resolve()
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    current_git = _git_state()
    errors = []
    if report.get("schema_version") != "exact-toy-audit/v1":
        errors.append("unsupported schema_version")
    if report.get("status") != "pass" or not all(report.get("checks", {}).values()):
        errors.append("toy audit did not pass every gate")
    if report.get("git", {}).get("commit") != current_git["commit"]:
        errors.append("toy audit commit does not match the current checkout")
    if not report.get("git", {}).get("tracked_clean", False):
        errors.append("toy audit was generated from a tracked-dirty checkout")
    if not current_git["tracked_clean"]:
        errors.append("current checkout has tracked changes")
    return {
        "status": "pass" if not errors else "no_go",
        "audit_path": str(audit_path),
        "commit": current_git["commit"],
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", help="Write a fresh stage-0 JSON report")
    mode.add_argument("--verify", help="Verify a saved report against the current Git commit")
    parser.add_argument("--base-distractors", type=int, default=2)
    parser.add_argument("--max-distractors", type=int, default=6)
    parser.add_argument("--monte-carlo-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if args.verify:
        report = verify_toy_audit(args.verify)
        printable = report
    else:
        report = run_toy_audit(
            base_distractors=args.base_distractors,
            max_distractors=args.max_distractors,
            monte_carlo_samples=args.monte_carlo_samples,
            seed=args.seed,
        )
        _write_json_atomic(Path(args.output), report)
        printable = {
            "status": report["status"],
            "output": str(Path(args.output).expanduser().resolve()),
            "git": report["git"],
            "checks": report["checks"],
            "maximum_conservation_error": report["maximum_conservation_error"],
        }
    print(json.dumps(printable, allow_nan=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
