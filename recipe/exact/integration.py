"""Thin adapters between EXACT and the vendored verl-agent framework."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from recipe.exact.env_probes import conservative_future_schema


class ExactEnvironmentAdapter:
    """Resolve optional EXACT capabilities without coupling environment bases."""

    def __init__(self, manager: Any):
        self.manager = manager

    def snapshots(self) -> list[Mapping[str, Any]]:
        provider = getattr(self.manager, "exact_credit_snapshots", None)
        if provider is None:
            provider = getattr(self.manager.envs, "exact_credit_snapshots", None)
        if provider is None:
            raise NotImplementedError(f"{type(self.manager).__name__} does not provide an EXACT credit probe")
        return list(provider())

    def effect_schemas(
        self,
        snapshots: Sequence[Mapping[str, Any]],
    ) -> list[Mapping[str, Any]]:
        provider = getattr(self.manager, "exact_effect_schemas", None)
        if provider is not None:
            schemas = list(provider(list(snapshots)))
        else:
            provider = getattr(self.manager.envs, "exact_effect_schemas", None)
            if provider is not None:
                schemas = list(provider())
            else:
                environment = str(self.manager.config.env.env_name).split("/")[0].lower()
                schemas = [conservative_future_schema(snapshot, environment) for snapshot in snapshots]
        if len(schemas) != len(snapshots):
            raise ValueError("EXACT snapshot/effect-schema batch size mismatch")
        return schemas

    def resolve_effect_schemas(
        self,
        schemas: Sequence[Mapping[str, Any]],
        text_actions: Sequence[str],
        response_token_ids: Any,
        response_mask: Any,
        tokenizer: Any,
    ) -> list[Mapping[str, Any]]:
        resolver = getattr(self.manager, "resolve_exact_effect_schemas", None)
        if resolver is None:
            return list(schemas)
        return list(
            resolver(
                schemas=list(schemas),
                text_actions=list(text_actions),
                response_token_ids=response_token_ids,
                response_mask=response_mask,
                tokenizer=tokenizer,
            )
        )
