"""Typed, serializable data contracts used by the EXACT credit compiler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np

ResourceSet = Tuple[str, ...]


def _stable_unique_strings(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} must contain unique values")
    if any(not value for value in result):
        raise ValueError(f"{field_name} must not contain empty values")
    return result


@dataclass(frozen=True)
class FactorSnapshot:
    """Programmatic verifier state at one environment checkpoint.

    ``values`` are training-only instrumentation and must never be inserted into
    the policy prompt. ``read_sets`` describes the state resources each factor
    may read; missing entries are treated as unknown rather than empty.
    """

    checkpoint_id: int
    factor_ids: tuple[str, ...]
    values: np.ndarray
    read_sets: Mapping[str, ResourceSet] = field(default_factory=dict)
    schema_version: str = "exact.credit.v1"

    def __post_init__(self) -> None:
        factor_ids = _stable_unique_strings(self.factor_ids, "factor_ids")
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1 or values.shape[0] != len(factor_ids):
            raise ValueError("values must be one-dimensional and aligned with factor_ids")
        if not np.all(np.isfinite(values)):
            raise ValueError("factor values must be finite")
        if self.checkpoint_id < 0:
            raise ValueError("checkpoint_id must be non-negative")

        normalized_read_sets = {}
        for factor_id, resources in self.read_sets.items():
            if factor_id not in factor_ids:
                raise ValueError(f"read set refers to unknown factor: {factor_id}")
            normalized_read_sets[factor_id] = _stable_unique_strings(resources, f"read_sets[{factor_id}]")

        object.__setattr__(self, "factor_ids", factor_ids)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "read_sets", normalized_read_sets)

    def value_by_id(self) -> Mapping[str, float]:
        return dict(zip(self.factor_ids, self.values.tolist()))


@dataclass(frozen=True)
class CreditAtom:
    """One return-conserving factor delta or the terminal residual."""

    atom_id: str
    value: float
    step_id: int | None
    factor_id: str | None
    read_set: ResourceSet = ()
    is_residual: bool = False

    def __post_init__(self) -> None:
        if not self.atom_id:
            raise ValueError("atom_id must not be empty")
        if not np.isfinite(self.value):
            raise ValueError("atom value must be finite")
        if self.is_residual and self.factor_id is not None:
            raise ValueError("the residual atom cannot have a factor_id")
        if not self.is_residual and (self.step_id is None or self.factor_id is None):
            raise ValueError("factor atoms require step_id and factor_id")
        object.__setattr__(self, "read_set", _stable_unique_strings(self.read_set, "read_set"))


@dataclass(frozen=True)
class EffectSpan:
    """A policy score span with a prefix-predictable effect signature."""

    span_id: str
    step_id: int
    token_start: int
    token_end: int
    bucket: str = "default"
    possible_write_set: ResourceSet = ()
    control_parents: ResourceSet = ()
    context_sources: ResourceSet = ()
    opaque: bool = False

    def __post_init__(self) -> None:
        if not self.span_id:
            raise ValueError("span_id must not be empty")
        if self.step_id <= 0:
            raise ValueError("step_id is one-based and must be positive")
        if self.token_start < 0 or self.token_end <= self.token_start:
            raise ValueError("span token offsets must define a non-empty half-open range")
        for name in ("possible_write_set", "control_parents", "context_sources"):
            object.__setattr__(self, name, _stable_unique_strings(getattr(self, name), name))


@dataclass(frozen=True)
class SpanRoute:
    """Certified conservative atom route for one effect span.

    ``descendant_atom_ids=None`` means that the span is opaque and therefore
    receives all atoms. Explicit routes are still forced to include the terminal
    residual by default in the compiler.
    """

    span: EffectSpan
    descendant_atom_ids: tuple[str, ...] | None = None
    soundness_certificate: str | None = None
    route_kind: str = "explicit"

    def __post_init__(self) -> None:
        if self.route_kind not in {"explicit", "resource_graph"}:
            raise ValueError(f"unsupported route kind: {self.route_kind}")
        if self.descendant_atom_ids is not None:
            object.__setattr__(
                self,
                "descendant_atom_ids",
                _stable_unique_strings(self.descendant_atom_ids, "descendant_atom_ids"),
            )


def snapshots_from_values(
    factor_ids: Sequence[str],
    checkpoint_values: Sequence[Sequence[float]],
    read_sets: Mapping[str, Sequence[str]] | None = None,
) -> tuple[FactorSnapshot, ...]:
    """Convenience constructor used by fixtures and simple environment probes."""

    normalized_read_sets = {
        factor_id: tuple(resources) for factor_id, resources in (read_sets or {}).items()
    }
    return tuple(
        FactorSnapshot(
            checkpoint_id=checkpoint_id,
            factor_ids=tuple(factor_ids),
            values=np.asarray(values, dtype=np.float64),
            read_sets=normalized_read_sets,
        )
        for checkpoint_id, values in enumerate(checkpoint_values)
    )
