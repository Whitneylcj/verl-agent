"""Source-checked ALFWorld PDDL semantics, without a planner or simulator.

The abstraction is a powerset of Boolean values for each ground predicate.
Joins intentionally forget correlations. This can retain extra edges, but cannot
certify an invariant by assuming a particular future policy, action or outcome.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from recipe.exact.credit_spec import FactorSnapshot

SCHEMA = "exact.alfworld.pddl.predicates.v1"
DOMAIN_PATH = Path(__file__).resolve().parents[2] / "agent_system/environments/env_package/alfworld/alfworld/data/alfred.pddl"
PROTECTABLE = frozenset({"isclean", "ishot", "iscool", "istoggled", "inreceptacle"})
SUPPORTED_DOMAIN_HASH = "decd886f9a13e537453b3d9eb8e391aac64fc434eae47d82e65c21ff50750bd9"


def parse_pddl(source: str) -> tuple:
    tokens = iter(re.findall(r"\(|\)|[^\s()]+", re.sub(r";[^\n]*", "", source).lower()))

    def read(token):
        if token != "(":
            if token == ")":
                raise ValueError("unexpected PDDL closing parenthesis")
            return token
        result = []
        for child in tokens:
            if child == ")":
                return tuple(result)
            result.append(read(child))
        raise ValueError("unclosed PDDL expression")

    result = read(next(tokens, ""))
    if next(tokens, None) is not None or not isinstance(result, tuple) or not result or result[0] != "define":
        raise ValueError("expected exactly one PDDL definition")
    return result


def content_hash(value: Any) -> str:
    return hashlib.sha256(repr(value).encode()).hexdigest()


def typed_names(items: tuple) -> tuple[tuple[str, str], ...]:
    result, pending = [], []
    index = 0
    while index < len(items):
        item = items[index]
        if not isinstance(item, str):
            raise ValueError("unsupported PDDL type expression")
        if item == "-":
            if not pending or index + 1 >= len(items):
                raise ValueError("invalid typed PDDL names")
            result.extend((name, items[index + 1].rstrip(",")) for name in pending)
            pending = []
            index += 2
        else:
            pending.append(item.rstrip(","))
            index += 1
    result.extend((name, "object") for name in pending)
    return tuple(result)


@dataclass(frozen=True, order=True)
class Fact:
    predicate: str
    arguments: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.predicate}({','.join(self.arguments)})"


def canonical_facts(values) -> frozenset[Fact]:
    positive, negative = set(), set()
    for value in values:
        if isinstance(value, Fact):
            fact = Fact(value.predicate.lower(), tuple(arg.lower() for arg in value.arguments))
        elif isinstance(value, dict):
            fact = Fact(str(value["predicate"]).lower(), tuple(str(x).lower() for x in value["arguments"]))
        else:
            fact = Fact(str(value.name).lower(), tuple(str(arg.name).lower() for arg in value.arguments))
        if fact.predicate.startswith("not_"):
            negative.add(Fact(fact.predicate[4:], fact.arguments))
        else:
            positive.add(fact)
    if positive & negative:
        raise ValueError("contradictory canonical PDDL facts")
    return frozenset(positive)


def conjunction(parts):
    parts = tuple(part for part in parts if part is not True)
    if False in parts:
        return False
    if not parts:
        return True
    return parts[0] if len(parts) == 1 else ("and", *parts)


def disjunction(parts):
    parts = tuple(part for part in parts if part is not False)
    if True in parts:
        return True
    if not parts:
        return False
    return parts[0] if len(parts) == 1 else ("or", *parts)


def read_facts(expr) -> frozenset[Fact]:
    if isinstance(expr, bool):
        return frozenset()
    if expr[0] == "atom":
        return frozenset({expr[1]})
    return frozenset(f for part in expr[1:] for f in read_facts(part))


def truth_values(expr, values: dict[Fact, frozenset[bool]]) -> frozenset[bool]:
    if isinstance(expr, bool):
        return frozenset({expr})
    if expr[0] == "atom":
        return values.get(expr[1], frozenset({False}))
    if expr[0] == "not":
        return frozenset(not value for value in truth_values(expr[1], values))
    parts = [truth_values(part, values) for part in expr[1:]]
    if expr[0] == "and":
        return frozenset(([True] if all(True in part for part in parts) else []) + ([False] if any(False in part for part in parts) else []))
    if expr[0] == "or":
        return frozenset(([True] if any(True in part for part in parts) else []) + ([False] if all(False in part for part in parts) else []))
    raise ValueError(f"unsupported Boolean operator {expr[0]}")


@dataclass(frozen=True)
class Effect:
    condition: Any
    fact: Fact
    value: bool


@dataclass(frozen=True)
class GroundAction:
    name: str
    bindings: tuple[tuple[str, str], ...]
    precondition: Any
    effects: tuple[Effect, ...]

    def blocked(self, protections: frozenset[Fact]) -> bool:
        # Conditional deletes also block: no optimistic use of today's guard.
        return any(not effect.value and effect.fact in protections for effect in self.effects)

    def apply(self, facts: frozenset[Fact]) -> frozenset[Fact]:
        values = {fact: frozenset({True}) for fact in facts}
        if truth_values(self.precondition, values) != frozenset({True}):
            raise ValueError("action precondition is false")
        fired = [effect for effect in self.effects if truth_values(effect.condition, values) == frozenset({True})]
        adds = {effect.fact for effect in fired if effect.value}
        deletes = {effect.fact for effect in fired if not effect.value}
        # PDDL simultaneous effects delete first, then add (notably goto
        # when start == end). Conditional guards all read the pre-state.
        return frozenset((facts - deletes) | adds)


class SemanticModel:
    def __init__(self, game_data: dict):
        domain = parse_pddl(game_data["pddl_domain"])
        if content_hash(domain) != SUPPORTED_DOMAIN_HASH:
            raise ValueError("unsupported ALFWorld domain: sparse certificates require the audited domain")
        self.domain_hash = content_hash(domain)
        problem = parse_pddl(game_data["pddl_problem"])
        self.problem_hash = content_hash(problem)
        self.source_revision = f"alfworld-pddl:{self.domain_hash};problem:{self.problem_hash}"
        sections = {node[0]: node[1:] for node in problem[1:] if isinstance(node, tuple)}
        if sections.get(":domain") != ("alfred",):
            raise ValueError("expected ALFRED problem domain")
        self.objects = dict(typed_names(sections[":objects"]))
        for node in domain:
            if isinstance(node, tuple) and node and node[0] == ":constants":
                self.objects.update(typed_names(node[1:]))
        self.by_type = {kind: tuple(name for name, typ in self.objects.items() if typ == kind) for kind in set(self.objects.values())}
        self.predicates = {}
        for node in domain:
            if isinstance(node, tuple) and node and node[0] == ":predicates":
                self.predicates = {pred[0]: len(typed_names(pred[1:])) for pred in node[1:]}
        self.initial = frozenset(self._fact(node, {}) for node in sections[":init"] if node and node[0] != "=")
        raw_actions = [node for node in domain if isinstance(node, tuple) and node and node[0] == ":action"]

        def written(expr):
            if expr[0] == "and":
                return set().union(*(written(part) for part in expr[1:]))
            if expr[0] == "when":
                return written(expr[2])
            if expr[0] == "not":
                return {expr[1][0]}
            if expr[0] == "increase":
                return set()
            return {expr[0]}

        self.dynamic = set().union(*(written(node[node.index(":effect") + 1]) for node in raw_actions))
        self.static = frozenset(fact for fact in self.initial if fact.predicate not in self.dynamic)
        static_index = {}
        for fact in self.static:
            static_index.setdefault(fact.predicate, []).append(fact.arguments)
        actions = []
        for node in raw_actions:
            params = dict(typed_names(node[node.index(":parameters") + 1]))
            pre = node[node.index(":precondition") + 1]
            eff = node[node.index(":effect") + 1]
            conjuncts = pre[1:] if pre and pre[0] == "and" else (pre,)
            joins = [part for part in conjuncts if part and part[0] in self.predicates and part[0] not in self.dynamic]
            joins.sort(key=lambda part: len(static_index.get(part[0], ())))
            bindings = [{}]
            for part in joins:
                next_bindings = []
                for binding in bindings:
                    for row in static_index.get(part[0], ()):
                        candidate = dict(binding)
                        for arg, value in zip(part[1:], row, strict=True):
                            if arg in params and self.objects[value] == params[arg] and candidate.get(arg, value) == value:
                                candidate[arg] = value
                            elif arg != value:
                                break
                        else:
                            next_bindings.append(candidate)
                bindings = next_bindings
            for partial in bindings:
                missing = [var for var in params if var not in partial]
                for values in itertools.product(*(self.by_type.get(params[var], ()) for var in missing)):
                    binding = {**partial, **dict(zip(missing, values, strict=True))}
                    condition = self._ground(pre, binding)
                    if condition is False:
                        continue
                    effects = tuple(self._effects(eff, binding))
                    actions.append(GroundAction(node[1], tuple(binding.items()), condition, effects))
                    if len(actions) > 100_000:
                        raise ValueError("ALFWorld grounding limit exceeded")
        self.actions = tuple(actions)
        self.goal = self._ground(sections[":goal"][0], {})
        goal_facts = read_facts(self.goal)
        support = set()
        for action in self.actions:
            if any(effect.fact in goal_facts for effect in action.effects):
                support.update(read_facts(action.precondition))
                for effect in action.effects:
                    if effect.fact in goal_facts:
                        support.update(read_facts(effect.condition))
        self.channels = tuple(sorted(goal_facts | support))
        if not self.channels or len(self.channels) > 20_000:
            raise ValueError("unsupported or empty ALFWorld goal channel schema")
        self.channel_kinds = {fact.key: "goal" if fact in goal_facts else "support" for fact in self.channels}
        self.universe = frozenset(self.channels) | frozenset(fact for action in self.actions for fact in read_facts(action.precondition)) | frozenset(effect.fact for action in self.actions for effect in action.effects)
        self.universe |= frozenset(fact for action in self.actions for effect in action.effects for fact in read_facts(effect.condition))

    def _fact(self, node, binding):
        if not node or node[0] not in self.predicates or len(node) - 1 != self.predicates[node[0]]:
            raise ValueError(f"unknown predicate or arity: {node}")
        args = tuple(binding.get(arg, arg) for arg in node[1:])
        if any(arg not in self.objects for arg in args):
            raise ValueError(f"unknown or unbound PDDL entity: {args}")
        return Fact(node[0], args)

    def _ground(self, node, binding):
        if not node:
            return True
        kind = node[0]
        if kind in {"and", "or"}:
            parts = []
            for part in node[1:]:
                grounded = self._ground(part, binding)
                if grounded is (False if kind == "and" else True):
                    return grounded
                parts.append(grounded)
            return conjunction(parts) if kind == "and" else disjunction(parts)
        if kind == "not":
            part = self._ground(node[1], binding)
            return not part if isinstance(part, bool) else ("not", part)
        if kind == "=":
            return binding.get(node[1], node[1]) == binding.get(node[2], node[2])
        if kind in {"exists", "forall"}:
            params = typed_names(node[1])
            parts = [self._ground(node[2], {**binding, **dict(zip((var for var, _ in params), values, strict=True))}) for values in itertools.product(*(self.by_type.get(typ, ()) for _, typ in params))]
            return disjunction(parts) if kind == "exists" else conjunction(parts)
        fact = self._fact(node, binding)
        return ("atom", fact) if fact.predicate in self.dynamic else fact in self.static

    def _effects(self, node, binding, condition=True):
        if node[0] == "and":
            for part in node[1:]:
                yield from self._effects(part, binding, condition)
        elif node[0] == "when":
            yield from self._effects(node[2], binding, conjunction((condition, self._ground(node[1], binding))))
        elif node[0] == "increase":
            if node[1] != ("total-cost",):
                raise ValueError("unsupported numeric state effect")
        else:
            positive = node[0] != "not"
            yield Effect(condition, self._fact(node if positive else node[1], binding), positive)

    def validate_state(self, facts: frozenset[Fact]) -> None:
        if frozenset(fact for fact in facts if fact.predicate not in self.dynamic) != self.static:
            raise ValueError("native static facts disagree with the loaded PDDL problem")
        for fact in facts:
            self._fact((fact.predicate, *fact.arguments), {})

    def goal_holds(self, facts: frozenset[Fact]) -> bool:
        return truth_values(self.goal, {fact: frozenset({True}) for fact in facts}) == frozenset({True})

    def reachable_values(self, facts: frozenset[Fact], protections: frozenset[Fact], horizon: int):
        if horizon < 0 or not protections <= facts or any(fact.predicate not in PROTECTABLE for fact in protections):
            raise ValueError("invalid reachability prefix or protections")
        values = {fact: frozenset({fact in facts}) for fact in self.universe}
        actions = [action for action in self.actions if not action.blocked(protections)]
        for _ in range(horizon):
            # Synchronous propagation: preconditions and conditional guards
            # always read the previous abstract state, never effects in this step.
            updates = dict(values)
            for action in actions:
                if True not in truth_values(action.precondition, values):
                    continue
                for effect in action.effects:
                    if True in truth_values(effect.condition, values):
                        updates[effect.fact] = updates[effect.fact] | {effect.value}
            if updates == values:
                break
            values = updates
        return values


def compile_semantics(game_data: dict) -> SemanticModel:
    return SemanticModel(game_data)


def snapshot(native_state: dict, semantic_model: SemanticModel) -> FactorSnapshot:
    facts = canonical_facts(native_state["_facts"])
    semantic_model.validate_state(facts)
    if semantic_model.goal_holds(facts) != bool(native_state["won"]):
        raise ValueError("compiled PDDL goal disagrees with native won")
    ids = tuple(fact.key for fact in semantic_model.channels)
    return FactorSnapshot(
        checkpoint_id=int(native_state.get("exact_step", 0)),
        factor_ids=ids,
        values=np.asarray([float(fact in facts) for fact in semantic_model.channels]),
        read_sets={key: (f"alfworld.fact:{key}",) for key in ids},
        schema_version=SCHEMA,
        channel_roles=dict.fromkeys(ids, "process_verifier"),
        potential_weights=np.full(len(ids), 1.0 / len(ids)),
        source_revision=semantic_model.source_revision,
    )
