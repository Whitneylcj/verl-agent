"""One native ALFWorld action per response, with explicit irreversible commits.

Certificates are derived from pre-action state and decoded token prefixes only.
All invalid continuations remain in the support as a native invalid command.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from recipe.exact.alfworld_semantics import PROTECTABLE, Fact, canonical_facts, compile_semantics, snapshot

INVALID_COMMAND = "__exact_rejected_action__"
OPERATIONS = {
    "goto": ("gotolocation", ("?r",)),
    "take": ("pickupobject", ("?o", "?r")),
    "put": ("putobject", ("?o", "?r")),
    "open": ("openobject", ("?r",)),
    "close": ("closeobject", ("?r",)),
    "heat": ("heatobject", ("?o", "?r")),
    "cool": ("coolobject", ("?o", "?r")),
    "clean": ("cleanobject", ("?o", "?r")),
    "toggle": ("toggleobject", ("?o",)),
    "slice": ("sliceobject", ("?co", "?ko")),
    "look": ("look", ()),
    "inventory": ("inventory", ()),
    "examine": ("examineobject", ("?o",)),
    "examine_receptacle": ("examinereceptacle", ("?r",)),
    "help": ("help", ()),
}
COMMIT_PREDICATES = {"heat": "ishot", "cool": "iscool", "clean": "isclean", "toggle": "istoggled", "put": "inreceptacle"}
NATIVE_SOURCE_BLOBS = {
    "textworld.envs.pddl.pddl": "a06b138f8b9b76c8c29bfbba5b9f1af0a24b9bf6",
    "textworld.envs.pddl.logic": "41c5183572d1aa89e927727fadacf46d333700fc",
}


def verify_native_engine() -> str:
    """Reject unreviewed engine implementations instead of certifying guesses."""
    import importlib

    hashes = []
    for name, expected in NATIVE_SOURCE_BLOBS.items():
        module = importlib.import_module(name)
        source = Path(inspect.getfile(module)).read_bytes()
        actual = hashlib.sha1(f"blob {len(source)}\0".encode() + source).hexdigest()
        if actual != expected:
            raise RuntimeError(f"unsupported native source {name}: {actual}, expected {expected}")
        hashes.append(actual)
    return ";".join(hashes)


@dataclass(frozen=True)
class ParsedAction:
    op: str
    args: tuple[str, ...]
    commit: bool


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate action field")
        result[key] = value
    return result


def parse_action(response: str) -> ParsedAction:
    record = json.loads(response, object_pairs_hook=_unique_pairs)
    if not isinstance(record, dict) or tuple(record) != ("op", "args", "commit"):
        raise ValueError("expected ordered op/args/commit JSON object")
    op, args, commit = record["op"], record["args"], record["commit"]
    if not isinstance(op, str) or op not in OPERATIONS or not isinstance(args, list) or len(args) != len(OPERATIONS[op][1]):
        raise ValueError("unsupported operation or argument count")
    if not all(isinstance(arg, str) and arg and arg == arg.strip().lower() for arg in args) or type(commit) is not bool:
        raise ValueError("arguments must be canonical display names and commit must be Boolean")
    if commit and op not in COMMIT_PREDICATES:
        raise ValueError("this operation cannot commit shared state")
    return ParsedAction(op, tuple(args), commit)


@dataclass(frozen=True)
class ExecutionDecision:
    command: str
    syntax_valid: bool
    execution_valid: bool
    reason: str
    commit_facts: tuple[Fact, ...] = ()
    expected_facts: frozenset[Fact] | None = None


def guard_action(parsed_action: ParsedAction, native_state: dict, protections: frozenset[Fact]) -> ExecutionDecision:
    """Resolve against native applicable actions, then enforce every protection."""
    session = native_state["exact_session"]
    rows = session.action_rows(native_state)
    matches = [row for row in rows if row["op"] == parsed_action.op and tuple(row["args"]) == parsed_action.args]
    if not matches:
        return ExecutionDecision(INVALID_COMMAND, True, False, "not_applicable")
    commands = {row["command"] for row in matches}
    if len(commands) != 1:
        raise ValueError("ambiguous native command grounding")
    facts = canonical_facts(native_state["_facts"])
    if any(row["action"].blocked(protections) for row in matches):
        return ExecutionDecision(INVALID_COMMAND, True, False, "protected_fact")
    next_states = {row["action"].apply(facts) for row in matches}
    if len(next_states) != 1:
        raise ValueError("native command aliases have different effects")
    expected = next_states.pop()
    commits = set()
    if parsed_action.commit:
        for row in matches:
            for effect in row["action"].effects:
                if effect.value and effect.fact.predicate == COMMIT_PREDICATES[parsed_action.op] and effect.fact in expected:
                    commits.add(effect.fact)
        if len(commits) != 1 or not all(fact.predicate in PROTECTABLE for fact in commits):
            raise ValueError("commit must identify one established object predicate")
    return ExecutionDecision(next(iter(commands)), True, True, "accepted", tuple(sorted(commits)), expected)


def _prefix_selection(prefix: str) -> tuple[str | None, tuple[str, ...]]:
    """Read only complete fields/arguments already present in the prefix.

    Do not validate the eventual complete response here. A later malformed
    suffix cannot change a certificate already issued for an earlier token.
    """
    decoder = json.JSONDecoder()
    match = re.match(r'\s*\{\s*"op"\s*:\s*', prefix)
    if match is None:
        return None, ()
    try:
        op, end = decoder.raw_decode(prefix, match.end())
    except ValueError:
        return None, ()
    if not isinstance(op, str) or op not in OPERATIONS:
        return None, ()
    rest = re.match(r'\s*,\s*"args"\s*:\s*\[\s*', prefix[end:])
    if rest is None:
        return op, ()
    position = end + rest.end()
    args = []
    while position < len(prefix):
        try:
            arg, end = decoder.raw_decode(prefix, position)
        except ValueError:
            break
        if not isinstance(arg, str):
            break
        args.append(arg)
        comma = re.match(r"\s*,\s*", prefix[end:])
        if comma is None:
            break
        position = end + comma.end()
    return op, tuple(args)


@dataclass(frozen=True)
class PrefixCertificate:
    source_hash: str
    state_hash: str
    prefix_length: int
    prefix_sha256: str
    horizon: int
    protections: tuple[str, ...]
    immediate_factor_ids: tuple[str, ...]
    future_factor_ids: tuple[str, ...]
    invariant_values: tuple[tuple[str, float], ...]
    reason: str = "boolean-powerset-reachability-with-permanent-write-guards-v1"


def compile_prefix_certificate(pre_state: dict, response_prefix: str, protections, horizon: int) -> PrefixCertificate:
    registry = pre_state
    if registry["kind"] != "alfworld-prefix-registry-v1" or horizon != registry["horizon"] or tuple(protections) != tuple(registry["protections"]):
        raise ValueError("prefix registry mismatch")
    op, args = _prefix_selection(response_prefix)
    candidates = [row for row in registry["actions"] if (op is None or row["op"] == op) and tuple(row["args"][: len(args)]) == args]
    changes = {key for row in candidates for key in row["changed_factor_ids"]}
    # Invalid continuations are identity transitions for all process predicates.
    return PrefixCertificate(
        source_hash=registry["source_hash"],
        state_hash=registry["state_hash"],
        prefix_length=len(response_prefix),
        horizon=horizon,
        prefix_sha256=hashlib.sha256(response_prefix.encode()).hexdigest(),
        protections=tuple(protections),
        immediate_factor_ids=tuple(key for key in registry["factor_ids"] if key in changes),
        future_factor_ids=tuple(registry["future_factor_ids"]),
        invariant_values=tuple((key, float(value)) for key, value in registry["invariant_values"]),
    )


class CodepointTokenizer:
    """Deterministic character scoring for CPU tests and native-only probes."""

    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)


def _prefix_decoder_supported(tokenizer) -> bool:
    if type(tokenizer) is CodepointTokenizer:
        return True
    # ByteLevel concatenates bytes. Only an unfinished UTF-8 suffix is unstable;
    # it is detected below. Arbitrary custom decoders get the broad envelope.
    decoder = getattr(getattr(tokenizer, "backend_tokenizer", None), "decoder", None)
    if decoder is None:
        return False
    try:
        return json.loads(decoder.__getstate__()).get("type") == "ByteLevel"
    except (AttributeError, TypeError, ValueError):
        return False


def resolve_effect_schema(registry: dict, token_ids, tokenizer) -> dict:
    if not token_ids:
        raise ValueError("cannot score an empty ALFWorld response")
    spans = []
    previous_signature = None
    prefix_decoder_supported = _prefix_decoder_supported(tokenizer)
    for start in range(len(token_ids)):
        prefix = tokenizer.decode(list(token_ids[:start]), skip_special_tokens=True, clean_up_tokenization_spaces=False) if prefix_decoder_supported else ""
        # A decoder with unstable byte prefixes gets the pre-response envelope.
        # This check uses only the current prefix, never any future tokens.
        if "\ufffd" in prefix:
            prefix = ""
        cert = compile_prefix_certificate(registry, prefix, registry["protections"], registry["horizon"])
        signature = (cert.immediate_factor_ids, cert.future_factor_ids)
        if signature == previous_signature:
            spans[-1]["token_end"] = start + 1
            continue
        spans.append(
            {
                "span_id": f"alfworld.prefix:{start}",
                "token_start": start,
                "token_end": start + 1,
                "bucket": "alfworld.prefix",
                "opaque": False,
                "route_kind": "alfworld_prefix",
                "certificate": asdict(cert),
            }
        )
        previous_signature = signature
    return {
        "kind": "alfworld-concrete-effect-v1",
        "spans": spans,
        "fixed_horizon": registry["fixed_horizon"],
        "source_revision": registry["source_hash"],
        "factor_ids": registry["factor_ids"],
        "pre_values": registry["pre_values"],
        "state_hash": registry["state_hash"],
        "protections": registry["protections"],
        "prefix_decoder_supported": prefix_decoder_supported,
    }


class AlfworldExactSession:
    """Lives inside the native TextWorld wrapper; no extra env.step calls."""

    def __init__(self, game_data: dict, horizon: int, guard_enabled: bool = True):
        if horizon <= 0:
            raise ValueError("ALFWorld horizon must be positive")
        started = time.perf_counter()
        self.model = compile_semantics(game_data)
        self.compile_seconds = time.perf_counter() - started
        self.horizon = horizon
        self.guard_enabled = guard_enabled
        self.protections = frozenset()
        self.step_id = 0
        self.state = None
        self.record = None
        self._action_index = {}
        for action in self.model.actions:
            key = (action.name, tuple(sorted(action.bindings)))
            self._action_index[key] = action

    def action_rows(self, state):
        infos = {str(key).lower(): value for key, value in state["_entity_infos"].items()}
        rows = []
        name_to_op = {name: op for op, (name, _) in OPERATIONS.items()}
        native_actions, commands = state["_valid_actions"], state["_valid_commands"]
        if len(native_actions) != len(commands):
            raise ValueError("native action/command metadata mismatch")
        for native, command in zip(native_actions, commands, strict=True):
            name = str(native.name).lower()
            if name not in name_to_op:
                raise ValueError(f"unsupported native ALFWorld operation {name}")
            binding = tuple(sorted(("?" + str(key.name).lstrip("?").lower(), str(value.name).lower()) for key, value in native.mapping.items()))
            action = self._action_index.get((name, binding))
            if action is None:
                raise ValueError("native action is absent from the compiled grounded domain")
            op = name_to_op[name]
            by_var = dict(binding)
            args = tuple(str(infos[by_var[var]].name).lower() for var in OPERATIONS[op][1])
            rows.append({"op": op, "args": args, "command": str(command), "action": action})
        return rows

    def capture(self, native_state: dict) -> dict:
        started = time.perf_counter()
        state = dict(native_state)
        state["exact_step"] = self.step_id
        typed = snapshot(state, self.model)
        facts = canonical_facts(state["_facts"])
        if not self.protections <= facts:
            raise ValueError("a protected predicate was destroyed")
        active = not bool(state["won"]) and self.step_id < self.horizon
        future = self.model.reachable_values(facts, self.protections, self.horizon - self.step_id if active else 0)
        invariant = {fact.key: float(fact in facts) for fact in self.model.channels if future[fact] == frozenset({fact in facts})}
        rows = self.action_rows(state) if active else []
        actions = []
        public_actions = []
        for row in rows:
            action = row["action"]
            if action.blocked(self.protections):
                continue
            after = action.apply(facts)
            changed = tuple(fact.key for fact in self.model.channels if (fact in facts) != (fact in after))
            actions.append({"op": row["op"], "args": row["args"], "changed_factor_ids": changed})
            public_actions.append({"op": row["op"], "args": list(row["args"]), "commit": False})
        ids = tuple(typed.factor_ids)
        from recipe.exact.alfworld_semantics import content_hash

        registry = {
            "kind": "alfworld-prefix-registry-v1",
            "source_hash": self.model.source_revision,
            "state_hash": content_hash(tuple(sorted(facts))),
            "factor_ids": ids,
            "pre_values": tuple(typed.values.tolist()),
            "actions": actions,
            "protections": tuple(sorted(fact.key for fact in self.protections)),
            "horizon": self.horizon - self.step_id,
            "fixed_horizon": self.horizon,
            "future_factor_ids": tuple(key for key in ids if key not in invariant),
            "invariant_values": tuple(sorted(invariant.items())),
        }
        record = asdict(typed)
        record["values"] = tuple(typed.values.tolist())
        record["potential_weights"] = tuple(typed.potential_weights.tolist())
        record["prefix_registry"] = registry
        record["channel_kinds"] = self.model.channel_kinds
        record["semantic_compile_seconds"] = self.compile_seconds if self.step_id == 0 else 0.0
        record["capture_seconds"] = time.perf_counter() - started
        self.state = state
        self.state["exact_session"] = self
        self.record = record
        self.public_actions = public_actions
        return record

    def prepare(self, response: str) -> ExecutionDecision:
        if self.state is None or self.step_id >= self.horizon:
            raise RuntimeError("ALFWorld session is inactive")
        try:
            parsed = parse_action(response)
        except (TypeError, ValueError):
            return ExecutionDecision(INVALID_COMMAND, False, False, "invalid_json_action")
        if not self.guard_enabled:
            parsed = ParsedAction(parsed.op, parsed.args, False)
        return guard_action(parsed, self.state, self.protections)

    def finish(self, decision: ExecutionDecision, native_state: dict) -> dict:
        before = canonical_facts(self.state["_facts"])
        after = canonical_facts(native_state["_facts"])
        expected = decision.expected_facts if decision.execution_valid else before
        if after != expected:
            raise ValueError("native transition disagrees with compiled PDDL effects")
        if decision.execution_valid:
            self.protections |= frozenset(decision.commit_facts)
        self.step_id += 1
        return self.capture(native_state)

    def public_protections(self) -> list[dict]:
        # Each fact was explicitly requested and established by the model's own
        # successful action. Never expose hidden goal matching or verifier values.
        infos = {str(key).lower(): value for key, value in self.state["_entity_infos"].items()}
        return [{"predicate": fact.predicate, "args": [str(infos[arg].name).lower() for arg in fact.arguments]} for fact in sorted(self.protections)]
