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
    channel_target_masses: Mapping[str, float]
    channel_conservation_errors: Mapping[str, float]

    @property
    def closure_atoms(self) -> tuple[CreditAtom, ...]:
        return tuple(atom for atom in self.atoms if atom.atom_kind == "closure")

    @property
    def opaque_target_atom(self) -> CreditAtom:
        atoms = [atom for atom in self.atoms if atom.atom_kind == "opaque_target"]
        if len(atoms) != 1:
            raise RuntimeError("a conserved atom set must contain exactly one opaque_target")
        return atoms[0]

    @property
    def closure_abs_mass(self) -> float:
        return float(sum(abs(atom.value) for atom in self.closure_atoms))

    @property
    def total_abs_mass(self) -> float:
        return float(sum(abs(atom.value) for atom in self.atoms))

    @property
    def closure_abs_ratio(self) -> float:
        return self.closure_abs_mass / self.total_abs_mass if self.total_abs_mass > 0 else 0.0

    @property
    def opaque_target_abs_ratio(self) -> float:
        opaque_abs = abs(self.opaque_target_atom.value)
        return opaque_abs / self.total_abs_mass if self.total_abs_mass > 0 else 0.0


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
    closure_route_density: float


def _validate_snapshots(snapshots: Sequence[FactorSnapshot]) -> tuple[FactorSnapshot, ...]:
    snapshots = tuple(snapshots)
    if len(snapshots) < 2:
        raise ValueError("EXACT requires an initial and at least one post-action checkpoint")
    factor_ids = snapshots[0].factor_ids
    schema_version = snapshots[0].schema_version
    potential_weights = snapshots[0].potential_weights
    channel_roles = snapshots[0].channel_roles
    read_sets = snapshots[0].read_sets
    source_revision = snapshots[0].source_revision
    for expected_checkpoint, snapshot in enumerate(snapshots):
        if snapshot.checkpoint_id != expected_checkpoint:
            raise ValueError("checkpoint IDs must be contiguous and start at zero")
        if snapshot.factor_ids != factor_ids:
            raise ValueError("factor IDs must remain stable within a trajectory")
        if snapshot.schema_version != schema_version:
            raise ValueError("factor schema version changed within a trajectory")
        if snapshot.channel_roles != channel_roles:
            raise ValueError("channel roles changed within a trajectory")
        if snapshot.read_sets != read_sets:
            raise ValueError("channel read sets changed within a trajectory")
        if snapshot.source_revision != source_revision:
            raise ValueError("verifier source revision changed within a trajectory")
        weights_changed = (potential_weights is None) != (snapshot.potential_weights is None)
        if potential_weights is not None and snapshot.potential_weights is not None:
            weights_changed = weights_changed or not np.array_equal(
                snapshot.potential_weights,
                potential_weights,
            )
        if weights_changed:
            raise ValueError("factor potential weights changed within a trajectory")
    return snapshots


def build_scoped_conserved_atoms(
    snapshots: Sequence[FactorSnapshot],
    episode_return: float,
    potential: IdentityPotential,
    tolerance: float = 1e-10,
) -> ConservedAtoms:
    """Construct channel deltas, local closures, and one opaque target.

    Return-component channels close to their native weighted end-to-end change.
    Process-verifier channels close to zero, so their shaping mass cannot alter
    the episode target. Any target mass not represented by official return
    components remains in the single opaque target atom.
    """

    snapshots = _validate_snapshots(snapshots)
    if not np.isfinite(episode_return):
        raise ValueError("episode_return must be finite")

    if potential.factor_ids != snapshots[0].factor_ids:
        raise ValueError("potential factor_ids do not match snapshot channels")
    has_return_components = any(role == "return_component" for role in snapshots[0].channel_roles.values())
    if has_return_components:
        native_weights = snapshots[0].potential_weights
        if native_weights is None:
            raise ValueError("return_component channels require native potential_weights")
        if not np.array_equal(potential.weights, native_weights):
            raise ValueError("return_component channels must use their exact native weights")

    terminal_step_id = len(snapshots)
    atoms: list[CreditAtom] = []
    channel_delta_sums = {channel_id: 0.0 for channel_id in potential.factor_ids}
    for step_id, (before, after) in enumerate(zip(snapshots[:-1], snapshots[1:]), start=1):
        deltas = potential.factor_values(after) - potential.factor_values(before)
        for channel_id, value in zip(before.factor_ids, deltas.tolist()):
            if channel_id in after.read_sets:
                read_set = after.read_sets[channel_id]
            elif channel_id in before.read_sets:
                read_set = before.read_sets[channel_id]
            else:
                read_set = (UNKNOWN_RESOURCE,)
            channel_delta_sums[channel_id] += float(value)
            atoms.append(
                CreditAtom(
                    atom_id=f"delta:{step_id}:{channel_id}",
                    atom_kind="delta",
                    channel_id=channel_id,
                    value=float(value),
                    step_id=step_id,
                    read_set=tuple(read_set),
                )
            )

    initial_values = potential.factor_values(snapshots[0])
    final_values = potential.factor_values(snapshots[-1])
    channel_target_masses: dict[str, float] = {}
    channel_conservation_errors: dict[str, float] = {}
    for channel_index, channel_id in enumerate(potential.factor_ids):
        role = snapshots[0].channel_roles[channel_id]
        target_mass = float(final_values[channel_index] - initial_values[channel_index]) if role == "return_component" else 0.0
        closure_value = float(target_mass - channel_delta_sums[channel_id])
        read_set = snapshots[-1].read_sets.get(
            channel_id,
            snapshots[0].read_sets.get(channel_id, (UNKNOWN_RESOURCE,)),
        )
        atoms.append(
            CreditAtom(
                atom_id=f"closure:{channel_id}",
                atom_kind="closure",
                channel_id=channel_id,
                value=closure_value,
                step_id=terminal_step_id,
                read_set=tuple(read_set),
            )
        )
        channel_target_masses[channel_id] = target_mass
        channel_error = abs(channel_delta_sums[channel_id] + closure_value - target_mass)
        channel_conservation_errors[channel_id] = float(channel_error)
        if channel_error > tolerance:
            raise AssertionError(f"channel conservation failed for {channel_id}: error={channel_error:.3e}, tolerance={tolerance:.3e}")

    opaque_target_value = float(episode_return - sum(channel_target_masses.values()))
    atoms.append(
        CreditAtom(
            atom_id="opaque_target",
            atom_kind="opaque_target",
            channel_id=None,
            value=opaque_target_value,
            step_id=terminal_step_id,
            read_set=("episode_return",),
        )
    )

    total = float(sum(atom.value for atom in atoms))
    error = abs(float(episode_return) - total)
    if error > tolerance:
        raise AssertionError(f"pathwise conservation failed: error={error:.3e}, tolerance={tolerance:.3e}")
    return ConservedAtoms(
        atoms=tuple(atoms),
        episode_return=float(episode_return),
        conservation_error=error,
        channel_target_masses=channel_target_masses,
        channel_conservation_errors=channel_conservation_errors,
    )


def temporal_routes(routes: Sequence[SpanRoute], atoms: Sequence[CreditAtom]) -> tuple[SpanRoute, ...]:
    """Replace route atom sets with the conservative temporal future cone."""

    opaque_ids = [atom.atom_id for atom in atoms if atom.atom_kind == "opaque_target"]
    if len(opaque_ids) != 1:
        raise ValueError("temporal routing requires exactly one opaque_target")
    result = []
    for route in routes:
        descendants = [atom.atom_id for atom in atoms if atom.atom_kind == "opaque_target" or atom.step_id >= route.span.step_id]
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
    unknown_kinds = {atom.atom_kind for atom in atoms} - {"delta", "closure", "opaque_target"}
    if unknown_kinds:
        raise ValueError(f"unsupported atom kinds: {sorted(unknown_kinds)}")
    all_atom_ids = tuple(atom_by_id)
    opaque_target_id = conserved_atoms.opaque_target_atom.atom_id
    closure_atom_ids = {atom.atom_id for atom in atoms if atom.atom_kind == "closure"}
    alpha_by_bucket = dict(alpha_by_bucket or {})

    span_credits = []
    descendant_pair_count = 0
    closure_descendant_pair_count = 0
    for route in routes:
        if route.span.opaque or route.descendant_atom_ids is None:
            descendant_ids = set(all_atom_ids)
        else:
            descendant_ids = set(route.descendant_atom_ids)
            unknown = descendant_ids - set(all_atom_ids)
            if unknown:
                raise ValueError(f"route {route.span.span_id} refers to unknown atoms: {sorted(unknown)}")
            descendant_ids.add(opaque_target_id)

        descendant_ids_ordered = tuple(atom_id for atom_id in all_atom_ids if atom_id in descendant_ids)
        non_descendant_ids = tuple(atom_id for atom_id in all_atom_ids if atom_id not in descendant_ids)
        causal_return = float(sum(atom_by_id[atom_id].value for atom_id in descendant_ids_ordered))
        control_return = float(sum(atom_by_id[atom_id].value for atom_id in non_descendant_ids))
        alpha = float(alpha_by_bucket.get(route.span.bucket, 0.0)) if mode == "graph_cv" else 0.0
        if not np.isfinite(alpha):
            raise ValueError(f"alpha for bucket {route.span.bucket} must be finite")
        credit = causal_return + alpha * control_return
        descendant_pair_count += len(descendant_ids_ordered)
        closure_descendant_pair_count += len(closure_atom_ids & descendant_ids)
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
    closure_denominator = len(routes) * len(closure_atom_ids)
    closure_route_density = closure_descendant_pair_count / closure_denominator if closure_denominator else 0.0
    return ExactCreditResult(
        span_credits=tuple(span_credits),
        conserved_atoms=conserved_atoms,
        cone_density=float(cone_density),
        closure_route_density=float(closure_route_density),
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
