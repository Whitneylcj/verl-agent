"""Pure credit-construction primitives for EXACT.

The functions in this module do not inspect language-model outputs or execute an
environment. They only transform already recorded, stop-gradient verifier state
and prefix-predictable routes into conserved span credits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from recipe.exact.appworld_schema import UNKNOWN_RESOURCE
from recipe.exact.credit_spec import CreditAtom, FactorSnapshot, SpanRoute


@dataclass(frozen=True)
class IdentityPotential:
    """Fixed additive potential ``sum_k weight[k] * z[k]``."""

    factor_ids: tuple[str, ...]
    weights: np.ndarray

    def __post_init__(self) -> None:
        factor_ids = tuple(self.factor_ids)
        if len(factor_ids) != len(set(factor_ids)):
            raise ValueError("potential factor_ids must be unique")
        if any(not factor_id for factor_id in factor_ids):
            raise ValueError("potential factor_ids must not contain empty values")
        weights = np.asarray(self.weights, dtype=np.float64)
        if len(factor_ids) == 0:
            raise ValueError("a potential requires at least one factor")
        if weights.shape != (len(factor_ids),):
            raise ValueError("potential weights must align with factor_ids")
        if not np.all(np.isfinite(weights)):
            raise ValueError("potential weights must be finite")
        object.__setattr__(self, "factor_ids", factor_ids)
        object.__setattr__(self, "weights", weights)

    @classmethod
    def normalized(cls, factor_ids: Sequence[str], scale: float = 1.0) -> IdentityPotential:
        factor_ids = tuple(factor_ids)
        if not factor_ids:
            raise ValueError("factor_ids must not be empty")
        return cls(factor_ids=factor_ids, weights=np.full(len(factor_ids), scale / len(factor_ids)))

    def factor_values(self, snapshot: FactorSnapshot) -> np.ndarray:
        if snapshot.factor_ids != self.factor_ids:
            raise ValueError("snapshot factor_ids do not match the fixed potential")
        return self.weights * snapshot.values

    def __call__(self, snapshot: FactorSnapshot) -> float:
        return float(np.sum(self.factor_values(snapshot)))


@dataclass(frozen=True)
class ConservedAtoms:
    atoms: tuple[CreditAtom, ...]
    episode_return: float
    conservation_error: float

    @property
    def residual(self) -> CreditAtom:
        residuals = [atom for atom in self.atoms if atom.is_residual]
        if len(residuals) != 1:
            raise RuntimeError("a conserved atom set must contain exactly one residual")
        return residuals[0]

    @property
    def residual_ratio(self) -> float:
        residual_abs = abs(self.residual.value)
        total_abs = sum(abs(atom.value) for atom in self.atoms)
        return residual_abs / total_abs if total_abs > 0 else 0.0


@dataclass(frozen=True)
class SpanCredit:
    span_id: str
    step_id: int
    bucket: str
    descendant_atom_ids: tuple[str, ...]
    non_descendant_atom_ids: tuple[str, ...]
    causal_return: float
    control_return: float
    alpha: float
    credit: float


@dataclass(frozen=True)
class ExactCreditResult:
    span_credits: tuple[SpanCredit, ...]
    conserved_atoms: ConservedAtoms
    cone_density: float


def _validate_snapshots(snapshots: Sequence[FactorSnapshot]) -> tuple[FactorSnapshot, ...]:
    snapshots = tuple(snapshots)
    if len(snapshots) < 2:
        raise ValueError("EXACT requires an initial and at least one post-action checkpoint")
    factor_ids = snapshots[0].factor_ids
    schema_version = snapshots[0].schema_version
    potential_weights = snapshots[0].potential_weights
    for expected_checkpoint, snapshot in enumerate(snapshots):
        if snapshot.checkpoint_id != expected_checkpoint:
            raise ValueError("checkpoint IDs must be contiguous and start at zero")
        if snapshot.factor_ids != factor_ids:
            raise ValueError("factor IDs must remain stable within a trajectory")
        if snapshot.schema_version != schema_version:
            raise ValueError("factor schema version changed within a trajectory")
        weights_changed = (potential_weights is None) != (snapshot.potential_weights is None)
        if potential_weights is not None and snapshot.potential_weights is not None:
            weights_changed = weights_changed or not np.array_equal(
                snapshot.potential_weights,
                potential_weights,
            )
        if weights_changed:
            raise ValueError("factor potential weights changed within a trajectory")
    return snapshots


def build_conserved_atoms(
    snapshots: Sequence[FactorSnapshot],
    episode_return: float,
    potential: IdentityPotential,
    tolerance: float = 1e-10,
) -> ConservedAtoms:
    """Construct factor deltas and the terminal residual for one trajectory."""

    snapshots = _validate_snapshots(snapshots)
    if not np.isfinite(episode_return):
        raise ValueError("episode_return must be finite")

    atoms = []
    for step_id, (before, after) in enumerate(zip(snapshots[:-1], snapshots[1:]), start=1):
        deltas = potential.factor_values(after) - potential.factor_values(before)
        for factor_id, value in zip(before.factor_ids, deltas.tolist()):
            if factor_id in after.read_sets:
                read_set = after.read_sets[factor_id]
            elif factor_id in before.read_sets:
                read_set = before.read_sets[factor_id]
            else:
                read_set = (UNKNOWN_RESOURCE,)
            atoms.append(
                CreditAtom(
                    atom_id=f"factor:{step_id}:{factor_id}",
                    value=float(value),
                    step_id=step_id,
                    factor_id=factor_id,
                    read_set=tuple(read_set),
                )
            )

    residual_value = float(episode_return - potential(snapshots[-1]) + potential(snapshots[0]))
    residual_read_set = sorted({resource for snapshot in snapshots for resources in snapshot.read_sets.values() for resource in resources} | {"episode_return"})
    atoms.append(
        CreditAtom(
            atom_id="residual",
            value=residual_value,
            step_id=None,
            factor_id=None,
            read_set=tuple(residual_read_set),
            is_residual=True,
        )
    )

    total = float(sum(atom.value for atom in atoms))
    error = abs(float(episode_return) - total)
    if error > tolerance:
        raise AssertionError(f"pathwise conservation failed: error={error:.3e}, tolerance={tolerance:.3e}")
    return ConservedAtoms(atoms=tuple(atoms), episode_return=float(episode_return), conservation_error=error)


def temporal_routes(routes: Sequence[SpanRoute], atoms: Sequence[CreditAtom]) -> tuple[SpanRoute, ...]:
    """Replace route atom sets with the conservative temporal future cone."""

    residual_id = next(atom.atom_id for atom in atoms if atom.is_residual)
    result = []
    for route in routes:
        descendants = [atom.atom_id for atom in atoms if atom.is_residual or (atom.step_id is not None and atom.step_id >= route.span.step_id)]
        if residual_id not in descendants:
            descendants.append(residual_id)
        result.append(
            SpanRoute(
                span=route.span,
                descendant_atom_ids=tuple(descendants),
                soundness_certificate="temporal-future-cone",
                route_kind="explicit",
            )
        )
    return tuple(result)


def compute_exact_credits(
    conserved_atoms: ConservedAtoms,
    routes: Sequence[SpanRoute],
    alpha_by_bucket: Mapping[str, float] | None = None,
    mode: str = "graph",
    force_residual_descendant: bool = True,
) -> ExactCreditResult:
    """Route conserved atoms to policy spans without changing their values."""

    if mode not in {"temporal", "graph", "graph_cv"}:
        raise ValueError(f"unsupported EXACT mode: {mode}")
    routes = tuple(routes)
    atoms = conserved_atoms.atoms
    if mode == "temporal":
        routes = temporal_routes(routes, atoms)

    atom_by_id = {atom.atom_id: atom for atom in atoms}
    if len(atom_by_id) != len(atoms):
        raise ValueError("atom IDs must be unique")
    all_atom_ids = tuple(atom_by_id)
    residual_id = conserved_atoms.residual.atom_id
    alpha_by_bucket = dict(alpha_by_bucket or {})

    span_credits = []
    descendant_pair_count = 0
    for route in routes:
        if route.span.opaque or route.descendant_atom_ids is None:
            descendant_ids = set(all_atom_ids)
        else:
            descendant_ids = set(route.descendant_atom_ids)
            unknown = descendant_ids - set(all_atom_ids)
            if unknown:
                raise ValueError(f"route {route.span.span_id} refers to unknown atoms: {sorted(unknown)}")
            if force_residual_descendant:
                descendant_ids.add(residual_id)

        descendant_ids_ordered = tuple(atom_id for atom_id in all_atom_ids if atom_id in descendant_ids)
        non_descendant_ids = tuple(atom_id for atom_id in all_atom_ids if atom_id not in descendant_ids)
        causal_return = float(sum(atom_by_id[atom_id].value for atom_id in descendant_ids_ordered))
        control_return = float(sum(atom_by_id[atom_id].value for atom_id in non_descendant_ids))
        alpha = float(alpha_by_bucket.get(route.span.bucket, 0.0)) if mode == "graph_cv" else 0.0
        if not np.isfinite(alpha):
            raise ValueError(f"alpha for bucket {route.span.bucket} must be finite")
        credit = causal_return + alpha * control_return
        descendant_pair_count += len(descendant_ids_ordered)
        span_credits.append(
            SpanCredit(
                span_id=route.span.span_id,
                step_id=route.span.step_id,
                bucket=route.span.bucket,
                descendant_atom_ids=descendant_ids_ordered,
                non_descendant_atom_ids=non_descendant_ids,
                causal_return=causal_return,
                control_return=control_return,
                alpha=alpha,
                credit=credit,
            )
        )

    denominator = len(routes) * len(atoms)
    cone_density = descendant_pair_count / denominator if denominator else 0.0
    return ExactCreditResult(
        span_credits=tuple(span_credits),
        conserved_atoms=conserved_atoms,
        cone_density=float(cone_density),
    )


class DelayedAlphaController:
    """Fit previous-batch non-descendant control coefficients.

    Inputs are detached gradient samples. ``hard_gradient_samples`` has shape
    ``[samples, dimensions]`` and ``control_gradient_samples`` has shape
    ``[samples, buckets, dimensions]``. Fitted coefficients are only exposed for
    the next batch, preserving predictability.
    """

    def __init__(
        self,
        bucket_names: Iterable[str],
        ridge: float = 1e-8,
        max_abs_alpha: float | None = None,
    ) -> None:
        self.bucket_names = tuple(bucket_names)
        if len(self.bucket_names) != len(set(self.bucket_names)):
            raise ValueError("bucket_names must be unique")
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        if max_abs_alpha is not None and max_abs_alpha <= 0:
            raise ValueError("max_abs_alpha must be positive when provided")
        self.ridge = float(ridge)
        self.max_abs_alpha = None if max_abs_alpha is None else float(max_abs_alpha)
        self._active_alpha: dict[str, float] = {bucket: 0.0 for bucket in self.bucket_names}
        self._pending_alpha: dict[str, float] | None = None

    @property
    def active(self) -> Mapping[str, float]:
        return dict(self._active_alpha)

    def fit_pending(self, hard_gradient_samples: np.ndarray, control_gradient_samples: np.ndarray) -> Mapping[str, float]:
        hard = np.asarray(hard_gradient_samples, dtype=np.float64)
        controls = np.asarray(control_gradient_samples, dtype=np.float64)
        if hard.ndim != 2:
            raise ValueError("hard_gradient_samples must have shape [samples, dimensions]")
        if controls.shape != (hard.shape[0], len(self.bucket_names), hard.shape[1]):
            raise ValueError("control_gradient_samples must have shape [samples, buckets, dimensions]")
        if not np.all(np.isfinite(hard)) or not np.all(np.isfinite(controls)):
            raise ValueError("gradient samples must be finite")

        centered_hard = hard - hard.mean(axis=0, keepdims=True)
        centered_controls = controls - controls.mean(axis=0, keepdims=True)
        sigma = np.einsum("sbd,scd->bc", centered_controls, centered_controls) / max(hard.shape[0], 1)
        covariance = np.einsum("sd,sbd->b", centered_hard, centered_controls) / max(hard.shape[0], 1)
        sigma = sigma + self.ridge * np.eye(len(self.bucket_names), dtype=np.float64)
        alpha = -np.linalg.pinv(sigma) @ covariance
        if self.max_abs_alpha is not None:
            alpha = np.clip(alpha, -self.max_abs_alpha, self.max_abs_alpha)
        self._pending_alpha = dict(zip(self.bucket_names, alpha.tolist()))
        return dict(self._pending_alpha)

    def advance(self) -> Mapping[str, float]:
        if self._pending_alpha is not None:
            self._active_alpha = self._pending_alpha
            self._pending_alpha = None
        return self.active

    def state_dict(self) -> Mapping[str, object]:
        return {
            "bucket_names": self.bucket_names,
            "ridge": self.ridge,
            "max_abs_alpha": self.max_abs_alpha,
            "active_alpha": dict(self._active_alpha),
            "pending_alpha": None if self._pending_alpha is None else dict(self._pending_alpha),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if tuple(state["bucket_names"]) != self.bucket_names:
            raise ValueError("delayed alpha checkpoint bucket names do not match this run")
        if float(state["ridge"]) != self.ridge or state["max_abs_alpha"] != self.max_abs_alpha:
            raise ValueError("delayed alpha checkpoint hyperparameters do not match this run")
        active = {str(key): float(value) for key, value in dict(state["active_alpha"]).items()}
        pending_state = state.get("pending_alpha")
        pending = None if pending_state is None else {str(key): float(value) for key, value in dict(pending_state).items()}
        if set(active) != set(self.bucket_names) or (pending is not None and set(pending) != set(self.bucket_names)):
            raise ValueError("delayed alpha checkpoint has an incompatible bucket schema")
        self._active_alpha = active
        self._pending_alpha = pending
