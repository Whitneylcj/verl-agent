"""Static AppWorld verifier graph and prefix-predictable effect schemas.

The task evaluator is training-only instrumentation. Nothing extracted here is
inserted into the policy prompt. Unknown evaluator/model patterns fail closed
to all known model resources.
"""

from __future__ import annotations

import ast
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any, Mapping, Sequence

from recipe.exact.appworld_adapter import parse_json_api_action

ALL_RESOURCE = "exact.resource.all"
UNKNOWN_RESOURCE = "exact.resource.unknown"
POLICY_CONTEXT_RESOURCE = "appworld.policy_context"
SUPPORTED_APPWORLD_SCHEMA_SOURCES = frozenset({("0.2.0.dev0", "a072b7a86e7c1d5b1d7175659d750ebb9b79f10a")})
_AUDITED_CROSS_APP_SERVICES = frozenset({"admin", "file_system", "gmail", "phone"})


def appworld_model_resource(app_name: str, model_name: str) -> str:
    return f"appworld.model:{app_name}.{model_name}"


def appworld_factor_id(requirement: str) -> str:
    digest = hashlib.sha256(requirement.strip().encode("utf-8")).hexdigest()[:16]
    return f"test:{digest}"


def detect_appworld_source_revision(module_file: str | Path) -> str | None:
    """Return the containing Git revision for an editable AppWorld install."""

    module_path = Path(module_file).expanduser().resolve()
    for parent in (module_path.parent, *module_path.parents):
        if not (parent / ".git").exists():
            continue
        try:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=parent,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=parent,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            return f"{revision}+dirty" if status else revision
        except (OSError, subprocess.CalledProcessError):
            return None
    return None


@dataclass(frozen=True)
class AppWorldFactorReads:
    read_sets: Mapping[str, tuple[str, ...]]
    opaque_factor_ids: tuple[str, ...]


@dataclass(frozen=True)
class _Facts:
    resources: frozenset[str] = frozenset()
    unresolved_models: bool = False

    def merge(self, *others: _Facts) -> _Facts:
        resources = set(self.resources)
        unresolved = self.unresolved_models
        for other in others:
            resources.update(other.resources)
            unresolved = unresolved or other.unresolved_models
        return _Facts(frozenset(resources), unresolved)


def _attribute_chain(node: ast.AST) -> tuple[str, ...] | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return tuple(reversed(parts))


def _literal_value(
    node: ast.AST,
    constants: Mapping[str, str],
    known_values: Mapping[str, Any],
) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in constants:
            return constants[node.id]
        return known_values.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _literal_value(node.value, constants, known_values)
        if base is None:
            return None
        if isinstance(base, Mapping):
            return base.get(node.attr)
        return getattr(base, node.attr, None)
    if isinstance(node, ast.IfExp):
        condition = _literal_value(node.test, constants, known_values)
        if isinstance(condition, bool):
            return _literal_value(
                node.body if condition else node.orelse,
                constants,
                known_values,
            )
        return None
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
        left = _literal_value(node.left, constants, known_values)
        right = _literal_value(node.comparators[0], constants, known_values)
        if left is None or right is None:
            return None
        if isinstance(node.ops[0], ast.Eq):
            return left == right
        if isinstance(node.ops[0], ast.NotEq):
            return left != right
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        operand = _literal_value(node.operand, constants, known_values)
        return None if not isinstance(operand, bool) else not operand
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_literal_value(item, constants, known_values) for item in node.elts]
        return None if any(value is None for value in values) else values
    if isinstance(node, ast.Dict):
        keys = [_literal_value(key, constants, known_values) for key in node.keys]
        values = [_literal_value(value, constants, known_values) for value in node.values]
        return None if any(item is None for item in (*keys, *values)) else dict(zip(keys, values))
    if isinstance(node, ast.Subscript):
        value = _literal_value(node.value, constants, known_values)
        key = _literal_value(node.slice, constants, known_values)
        try:
            return value[key]
        except (KeyError, IndexError, TypeError):
            return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_value(node.left, constants, known_values)
        right = _literal_value(node.right, constants, known_values)
        try:
            return left + right
        except TypeError:
            return None
    if isinstance(node, ast.JoinedStr):
        pieces: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                pieces.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                resolved = _literal_value(value.value, constants, known_values)
                if resolved is None:
                    return None
                pieces.append(str(resolved))
            else:
                return None
        return "".join(pieces)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        base = _literal_value(node.func.value, constants, known_values)
        if node.func.attr in {"title", "lower", "upper", "strip"} and isinstance(base, str):
            return getattr(base, node.func.attr)()
        if node.func.attr == "join" and isinstance(base, str) and len(node.args) == 1:
            values = _literal_value(node.args[0], constants, known_values)
            if isinstance(values, list) and all(isinstance(value, str) for value in values):
                return base.join(values)
    return None


def _constant_string(
    node: ast.AST,
    constants: Mapping[str, str],
    known_values: Mapping[str, Any],
) -> str | None:
    value = _literal_value(node, constants, known_values)
    return value if isinstance(value, str) else None


class _EvaluatorAnalyzer:
    def __init__(
        self,
        all_resources: Sequence[str],
        known_values: Mapping[str, Any],
    ) -> None:
        self.all_resources = frozenset(all_resources)
        self.known_values = dict(known_values)
        self.read_sets: dict[str, tuple[str, ...]] = {}
        self.opaque_factor_ids: set[str] = set()

    def _all(self) -> _Facts:
        return _Facts(self.all_resources, True)

    def _known_all(self) -> _Facts:
        return _Facts(self.all_resources, False)

    def expression(
        self,
        node: ast.AST | None,
        facts_by_name: Mapping[str, _Facts],
        constants: Mapping[str, str],
    ) -> _Facts:
        if node is None:
            return _Facts()
        if isinstance(node, ast.Name):
            if node.id == "models":
                return _Facts(unresolved_models=True)
            return facts_by_name.get(node.id, _Facts())
        if isinstance(node, ast.Attribute):
            chain = _attribute_chain(node)
            if (
                chain
                and len(chain) >= 4
                and chain[:2]
                in {
                    ("models", "start"),
                    ("models", "end"),
                }
            ):
                resource = appworld_model_resource(chain[2], chain[3])
                return _Facts(frozenset({resource})) if resource in self.all_resources else self._all()
            base = self.expression(node.value, facts_by_name, constants)
            return self._all() if base.unresolved_models else base
        if isinstance(node, ast.Call):
            chain = _attribute_chain(node.func)
            if chain in {
                ("models", "changed_records"),
                ("models", "changed_fields"),
                ("models", "changed_field_names"),
            }:
                model_name = _constant_string(node.args[0], constants, self.known_values) if node.args else None
                if model_name and "." in model_name:
                    app_name, short_model_name = model_name.split(".", 1)
                    resource = appworld_model_resource(app_name, short_model_name)
                    return _Facts(frozenset({resource})) if resource in self.all_resources else self._all()
                return self._all()
            if chain == ("models", "changed_model_names"):
                return self._known_all()
        if isinstance(node, ast.Lambda):
            return self.expression(node.body, facts_by_name, constants)

        merged = _Facts()
        for child in ast.iter_child_nodes(node):
            merged = merged.merge(self.expression(child, facts_by_name, constants))
        return self._all() if merged.unresolved_models else merged

    @staticmethod
    def _bind_target(
        target: ast.AST,
        facts: _Facts,
        constant: str | None,
        facts_by_name: dict[str, _Facts],
        constants: dict[str, str],
    ) -> None:
        if isinstance(target, ast.Name):
            facts_by_name[target.id] = facts
            if constant is None:
                constants.pop(target.id, None)
            else:
                constants[target.id] = constant
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                _EvaluatorAnalyzer._bind_target(element, facts, None, facts_by_name, constants)

    @staticmethod
    def _merge_scopes(
        scopes: Sequence[tuple[dict[str, _Facts], dict[str, str]]],
    ) -> tuple[dict[str, _Facts], dict[str, str]]:
        names = set().union(*(scope[0] for scope in scopes))
        merged_facts: dict[str, _Facts] = {}
        for name in names:
            facts = _Facts()
            for facts_by_name, _ in scopes:
                facts = facts.merge(facts_by_name.get(name, _Facts()))
            merged_facts[name] = facts
        common_constants: dict[str, str] = {}
        constant_names = set.intersection(*(set(scope[1]) for scope in scopes))
        for name in constant_names:
            values = {scope[1][name] for scope in scopes}
            if len(values) == 1:
                common_constants[name] = values.pop()
        return merged_facts, common_constants

    @staticmethod
    def _test_requirement(node: ast.With) -> str | None:
        if len(node.items) != 1 or not isinstance(node.items[0].context_expr, ast.Call):
            return None
        call = node.items[0].context_expr
        if not isinstance(call.func, ast.Name) or call.func.id != "test" or not call.args:
            return None
        value = call.args[0]
        return dedent(value.value).strip() if isinstance(value, ast.Constant) and isinstance(value.value, str) else None

    def statements(
        self,
        statements: Sequence[ast.stmt],
        facts_by_name: dict[str, _Facts],
        constants: dict[str, str],
    ) -> _Facts:
        total = _Facts()
        for statement in statements:
            requirement = self._test_requirement(statement) if isinstance(statement, ast.With) else None
            if requirement is not None:
                block_facts = self.statements(statement.body, facts_by_name, constants)
                factor_id = appworld_factor_id(requirement)
                resources = self.all_resources if block_facts.unresolved_models else block_facts.resources
                if factor_id in self.read_sets:
                    raise ValueError(f"duplicate AppWorld evaluator requirement: {requirement}")
                self.read_sets[factor_id] = tuple(sorted(resources))
                if block_facts.unresolved_models:
                    self.opaque_factor_ids.add(factor_id)
                total = total.merge(block_facts)
                continue

            if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                value = statement.value
                facts = self.expression(value, facts_by_name, constants)
                constant = _constant_string(value, constants, self.known_values)
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                for target in targets:
                    self._bind_target(target, facts, constant, facts_by_name, constants)
                total = total.merge(facts)
            elif isinstance(statement, ast.AugAssign):
                facts = self.expression(statement.value, facts_by_name, constants).merge(self.expression(statement.target, facts_by_name, constants))
                self._bind_target(statement.target, facts, None, facts_by_name, constants)
                total = total.merge(facts)
            elif isinstance(statement, (ast.If, ast.IfExp)):
                condition = self.expression(statement.test, facts_by_name, constants)
                branch_scopes = []
                literal_condition = _literal_value(
                    statement.test,
                    constants,
                    self.known_values,
                )
                branches = (statement.body,) if literal_condition is True else (statement.orelse,) if literal_condition is False else (statement.body, statement.orelse)
                for branch in branches:
                    branch_facts = dict(facts_by_name)
                    branch_constants = dict(constants)
                    total = total.merge(self.statements(branch, branch_facts, branch_constants))
                    branch_scopes.append((branch_facts, branch_constants))
                facts_by_name.clear()
                constants.clear()
                merged_facts, merged_constants = self._merge_scopes(branch_scopes)
                facts_by_name.update(merged_facts)
                constants.update(merged_constants)
                total = total.merge(condition)
            elif isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
                loop_facts = self.expression(
                    statement.iter if isinstance(statement, (ast.For, ast.AsyncFor)) else statement.test,
                    facts_by_name,
                    constants,
                )
                body_facts = dict(facts_by_name)
                body_constants = dict(constants)
                if isinstance(statement, (ast.For, ast.AsyncFor)):
                    self._bind_target(
                        statement.target,
                        loop_facts,
                        None,
                        body_facts,
                        body_constants,
                    )
                total = total.merge(
                    loop_facts,
                    self.statements(statement.body, body_facts, body_constants),
                    self.statements(statement.orelse, body_facts, body_constants),
                )
                merged_facts, merged_constants = self._merge_scopes([(dict(facts_by_name), dict(constants)), (body_facts, body_constants)])
                facts_by_name.clear()
                facts_by_name.update(merged_facts)
                constants.clear()
                constants.update(merged_constants)
            elif isinstance(statement, ast.Try):
                scopes = []
                for branch in [statement.body, *(handler.body for handler in statement.handlers)]:
                    branch_facts = dict(facts_by_name)
                    branch_constants = dict(constants)
                    total = total.merge(self.statements(branch, branch_facts, branch_constants))
                    scopes.append((branch_facts, branch_constants))
                merged_facts, merged_constants = self._merge_scopes(scopes)
                facts_by_name.clear()
                facts_by_name.update(merged_facts)
                constants.clear()
                constants.update(merged_constants)
                total = total.merge(
                    self.statements(statement.orelse, facts_by_name, constants),
                    self.statements(statement.finalbody, facts_by_name, constants),
                )
            elif isinstance(statement, (ast.With, ast.AsyncWith)):
                for item in statement.items:
                    total = total.merge(self.expression(item.context_expr, facts_by_name, constants))
                total = total.merge(self.statements(statement.body, facts_by_name, constants))
            else:
                total = total.merge(self.expression(statement, facts_by_name, constants))
        return total


def compile_appworld_factor_reads(
    evaluation_code: str,
    all_model_resources: Sequence[str],
    known_values: Mapping[str, Any] | None = None,
) -> AppWorldFactorReads:
    """Compile each ``with test(...)`` block to a conservative model read set."""

    module = ast.parse(evaluation_code)
    functions = [node for node in module.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "evaluate"]
    if len(functions) != 1:
        raise ValueError("AppWorld evaluator must contain exactly one evaluate function")
    analyzer = _EvaluatorAnalyzer(
        tuple(sorted(set(all_model_resources))),
        known_values or {},
    )
    analyzer.statements(functions[0].body, {}, {})
    if not analyzer.read_sets:
        raise ValueError("AppWorld evaluator contains no static test blocks")
    return AppWorldFactorReads(
        read_sets=dict(analyzer.read_sets),
        opaque_factor_ids=tuple(sorted(analyzer.opaque_factor_ids)),
    )


def build_appworld_effect_registry(
    api_docs: Mapping[str, Mapping[str, Any]],
    app_to_model_names: Mapping[str, Sequence[str]],
    appworld_version: str,
    appworld_source_revision: str | None = None,
) -> dict[str, Any]:
    """Build a version-gated, task-visible API-to-possible-writes registry."""

    app_resources = {app_name: tuple(sorted(appworld_model_resource(app_name, model_name) for model_name in model_names)) for app_name, model_names in app_to_model_names.items()}
    all_resources = tuple(sorted({item for values in app_resources.values() for item in values}))
    if not all_resources:
        all_resources = (ALL_RESOURCE,)
    version_supported = appworld_version in {version for version, _ in SUPPORTED_APPWORLD_SCHEMA_SOURCES}
    source_supported = (
        appworld_version,
        appworld_source_revision,
    ) in SUPPORTED_APPWORLD_SCHEMA_SOURCES
    shared_resources = {item for app_name in _AUDITED_CROSS_APP_SERVICES for item in app_resources.get(app_name, ())}
    api_possible_write_sets = {}
    for app_name, api_name_to_doc in api_docs.items():
        if source_supported:
            possible_writes = set(app_resources.get(app_name, ())) | shared_resources
            if not possible_writes:
                possible_writes = set(all_resources)
        else:
            possible_writes = set(all_resources)
        for api_name in api_name_to_doc:
            api_possible_write_sets[f"{app_name}.{api_name}"] = tuple(sorted(possible_writes))
    return {
        "kind": "appworld-prefix-effect-v1",
        "appworld_version": appworld_version,
        "appworld_source_revision": appworld_source_revision,
        "version_supported": version_supported,
        "source_supported": source_supported,
        "all_model_resources": all_resources,
        "selector_possible_write_set": all_resources,
        "api_possible_write_sets": api_possible_write_sets,
        "certificate": ("appworld-0.2.0.dev0-a072b7a-own-app-plus-audited-service-closure" if source_supported else "unsupported-source-global-write-fallback"),
    }


def _decode_prefix_boundary(
    tokenizer: Any,
    response_token_ids: Sequence[int],
    full_text: str,
    character_boundary: int,
) -> int | None:
    decode_kwargs = {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
    decoded_full = tokenizer.decode(list(response_token_ids), **decode_kwargs)
    if decoded_full != full_text:
        return None
    for token_end in range(1, len(response_token_ids) + 1):
        prefix = tokenizer.decode(list(response_token_ids[:token_end]), **decode_kwargs)
        if not full_text.startswith(prefix):
            return None
        if len(prefix) >= character_boundary:
            return token_end
    return None


def resolve_appworld_effect_schema(
    registry: Mapping[str, Any],
    text_action: str,
    response_token_ids: Sequence[int],
    tokenizer: Any,
) -> dict[str, Any]:
    """Resolve selector/argument spans using only information in each prefix."""

    if registry.get("kind") != "appworld-prefix-effect-v1":
        raise ValueError("unexpected AppWorld effect registry kind")
    valid_length = len(response_token_ids)
    if valid_length <= 0:
        raise ValueError("cannot resolve an empty AppWorld response")
    selector_writes = tuple(registry["selector_possible_write_set"])
    certificate = str(registry["certificate"])
    parsed = None
    try:
        parsed = parse_json_api_action(text_action)
    except (TypeError, ValueError):
        pass

    argument_start = None
    api_writes = selector_writes
    if parsed is not None and parsed.arguments_char_start is not None:
        argument_start = _decode_prefix_boundary(
            tokenizer,
            response_token_ids,
            text_action,
            parsed.arguments_char_start,
        )
        api_writes = tuple(registry["api_possible_write_sets"].get(f"{parsed.app}.{parsed.api}", selector_writes))

    common = {
        "route_kind": "resource_graph",
        "opaque": False,
        "context_sources": (POLICY_CONTEXT_RESOURCE,),
    }
    selector_end = argument_start if argument_start not in (None, valid_length) else valid_length
    selector = {
        **common,
        "span_id": "appworld.selector",
        "bucket": "appworld.selector",
        "token_start": 0,
        "token_end": selector_end,
        "possible_write_set": (*selector_writes, POLICY_CONTEXT_RESOURCE),
        "certificate": f"{certificate}:selector-union",
    }
    spans = [selector]
    if argument_start is not None and 0 < argument_start < valid_length:
        spans.append(
            {
                **common,
                "span_id": "appworld.arguments",
                "bucket": "appworld.arguments",
                "token_start": argument_start,
                "token_end": valid_length,
                "possible_write_set": (*api_writes, POLICY_CONTEXT_RESOURCE),
                "control_parents": ("appworld.selector",),
                "certificate": f"{certificate}:selected-api-prefix",
            }
        )
    return {
        "kind": "appworld-concrete-effect-v1",
        "spans": spans,
        "resolution_fallback": len(spans) == 1,
        "version_supported": bool(registry["version_supported"]),
        "source_supported": bool(registry["source_supported"]),
        "factor_count": int(registry.get("factor_count", 0)),
        "opaque_factor_count": int(registry.get("opaque_factor_count", 0)),
        "factor_schema_compile_fallback": bool(registry.get("factor_schema_compile_fallback", False)),
    }
