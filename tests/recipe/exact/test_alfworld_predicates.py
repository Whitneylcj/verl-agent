import copy

import numpy as np
import pytest

from recipe.exact.alfworld_adapter import compile_prefix_certificate, resolve_effect_schema
from recipe.exact.alfworld_semantics import canonical_facts, compile_semantics, snapshot, truth_values
from tests.recipe.exact.alfworld_fixtures import TASKS, CharacterTokenizer, execute, fact, game_data, native_state, response, session


@pytest.mark.parametrize("task", TASKS)
def test_full_official_goals_and_stable_process_channels(task):
    model = compile_semantics(game_data(task))
    initial = model.initial
    assert not model.goal_holds(initial)
    target = initial | {fact("inReceptacle", "apple1", "fridge")}
    predicates = {TASKS[1]: "isClean", TASKS[2]: "isHot", TASKS[3]: "isCool"}
    if task in predicates:
        assert not model.goal_holds(target)
        assert not model.goal_holds(target | {fact(predicates[task], "apple2")})
        target |= {fact(predicates[task], "apple1")}
    if task == TASKS[4]:
        assert not model.goal_holds(target)
        assert not model.goal_holds(target | {fact("inReceptacle", "apple2", "fridge2")})
        target |= {fact("inReceptacle", "apple2", "fridge")}
    if task == TASKS[5]:
        target = initial | {fact("holds", "a", "apple1"), fact("holdsAny", "a")}
        assert not model.goal_holds(target | {fact("isOn", "lamp")})
        target |= {fact("isToggled", "lamp")}
        assert not model.goal_holds(target - {fact("atLocation", "a", "lt")})
    assert model.goal_holds(target)
    before = snapshot(native_state(model, initial), model)
    after = snapshot(native_state(model, target), model)
    assert before.factor_ids == after.factor_ids
    assert all(f.predicate in model.dynamic for f in model.channels)
    assert all(role == "process_verifier" for role in before.channel_roles.values())
    assert np.sum(before.potential_weights) == pytest.approx(1)
    bad = native_state(model, target)
    bad["won"] = False
    with pytest.raises(ValueError, match="won"):
        snapshot(bad, model)


def test_temperature_cross_writes_and_guard_does_not_lock_shared_state():
    current = session()
    assert execute(current, "take", "apple1", "table").execution_valid
    assert execute(current, "goto", "micro").execution_valid
    assert execute(current, "heat", "apple1", "micro", commit=True).execution_valid
    assert current.protections == {fact("isHot", "apple1")}
    assert execute(current, "goto", "fridge").execution_valid
    rejected = execute(current, "cool", "apple1", "fridge")
    assert rejected.reason == "protected_fact"
    assert fact("isHot", "apple1") in current.state["_facts"]
    assert fact("isCool", "apple1") not in current.state["_facts"]
    assert current.step_id == 5
    assert execute(current, "open", "fridge").execution_valid
    assert execute(current, "put", "apple1", "fridge").execution_valid
    assert execute(current, "close", "fridge").execution_valid


def test_heat_cool_without_guard_and_conditional_toggle():
    current = session(guard=False)
    execute(current, "take", "apple1", "table")
    execute(current, "goto", "micro")
    execute(current, "heat", "apple1", "micro", commit=True)
    execute(current, "goto", "fridge")
    assert execute(current, "cool", "apple1", "fridge", commit=True).execution_valid
    assert fact("isCool", "apple1") in current.state["_facts"]
    assert fact("isHot", "apple1") not in current.state["_facts"]
    assert not current.protections
    execute(current, "goto", "table")
    execute(current, "toggle", "lamp", commit=True)
    assert fact("isOn", "lamp") in current.state["_facts"]
    execute(current, "toggle", "lamp", commit=True)
    assert fact("isOn", "lamp") not in current.state["_facts"]
    assert fact("isToggled", "lamp") in current.state["_facts"]


def test_wrong_placement_commit_freezes_temperature_through_holding():
    current = session()
    execute(current, "take", "apple1", "table")
    execute(current, "put", "apple1", "table", commit=True)
    assert not current.state["won"]  # Explicitly preserve bad model commitments.
    registry = current.record["prefix_registry"]
    invariants = dict(registry["invariant_values"])
    assert invariants["ishot(apple1)"] == 0
    assert invariants["inreceptacle(apple1,fridge)"] == 0
    assert "ishot(apple2)" in registry["future_factor_ids"]
    assert execute(current, "take", "apple1", "table").reason == "protected_fact"
    assert execute(current, "take", "apple2", "table").execution_valid
    assert current.public_protections() == [{"predicate": "inreceptacle", "args": ["apple1", "table"]}]


def test_natural_clean_monotonicity_and_future_unlocking():
    current = session(TASKS[1])
    assert "isclean(apple1)" in current.record["prefix_registry"]["future_factor_ids"]
    execute(current, "take", "apple1", "table")
    execute(current, "goto", "sink")
    execute(current, "clean", "apple1", "sink")
    assert not current.protections
    assert dict(current.record["prefix_registry"]["invariant_values"])["isclean(apple1)"] == 1


@pytest.mark.parametrize("bad", ["", "look", "{}", '{"op":"look","args":[],"commit":true}', '{"args":[],"op":"look","commit":false}', '{"op":"look","op":"look","args":[],"commit":false}', '{"op":"put","args":["apple1","table"],"commit":1}'])
def test_invalid_responses_consume_step_without_commit(bad):
    current = session()
    decision = current.prepare(bad)
    assert not decision.syntax_valid
    initial = current.state["_facts"]
    current.finish(decision, native_state(current.model, initial))
    assert current.step_id == 1 and not current.protections
    assert current.state["_facts"] == initial


def test_failed_commit_and_native_mismatch_stop_before_protection():
    current = session()
    assert not execute(current, "heat", "apple1", "micro", commit=True).execution_valid
    assert not current.protections
    decision = current.prepare(response("take", "apple1", "table"))
    with pytest.raises(ValueError, match="native transition"):
        current.finish(decision, native_state(current.model, current.model.initial))
    assert not current.protections


def test_prefix_certificates_do_not_look_at_suffix_or_new_commit():
    current = session()
    execute(current, "take", "apple1", "table")
    registry = copy.deepcopy(current.record["prefix_registry"])
    left = response("put", "apple1", "table", commit=True)
    right = response("put", "apple1", "table", commit=False)
    common = left[: left.index("true")]
    a = resolve_effect_schema(registry, list(map(ord, left)), CharacterTokenizer())
    b = resolve_effect_schema(registry, list(map(ord, right)), CharacterTokenizer())

    def at(schema, index):
        return next(r["certificate"] for r in schema["spans"] if r["token_start"] <= index < r["token_end"])

    for i in range(len(common) + 1):
        assert at(a, i) == at(b, i)
    for text in (left, right, left + "garbage", "malformed"):
        schema = resolve_effect_schema(registry, list(map(ord, text)), CharacterTokenizer())
        assert sum(s["token_end"] - s["token_start"] for s in schema["spans"]) == len(text)
        assert all(not s["certificate"]["protections"] for s in schema["spans"])
        assert all("ishot(apple1)" in s["certificate"]["future_factor_ids"] for s in schema["spans"])
    assert compile_prefix_certificate(registry, common, (), registry["horizon"]).immediate_factor_ids
    assert not compile_prefix_certificate(registry, '{"op":"inventory"', (), registry["horizon"]).immediate_factor_ids


def test_all_enumerated_reachable_states_respect_invariance_certificates():
    current = session(horizon=6)
    execute(current, "take", "apple1", "table")
    execute(current, "put", "apple1", "table", commit=True)
    initial = current.state["_facts"]
    values = current.model.reachable_values(initial, current.protections, 4)
    frontier, seen = {initial}, {initial}
    for _ in range(4):
        next_states = set()
        for state in frontier:
            concrete = {f: frozenset({True}) for f in state}
            for action in current.model.actions:
                if not action.blocked(current.protections) and True in truth_values(action.precondition, concrete):
                    next_states.add(action.apply(state))
        for state in next_states:
            for channel in current.model.channels:
                assert (channel in state) in values[channel]
        frontier = next_states - seen
        seen |= next_states
    assert len(seen) > 100


def test_sources_negative_facts_entity_renaming_and_zero_horizon():
    current = session()
    model = current.model
    changed = game_data()
    changed["pddl_domain"] = changed["pddl_domain"].replace("(isHot ?o)", "(isClean ?o)")
    with pytest.raises(ValueError, match="unsupported ALFWorld domain"):
        compile_semantics(changed)
    renamed = game_data()
    renamed["pddl_problem"] = renamed["pddl_problem"].replace("apple1", "apple99")
    renamed_model = compile_semantics(renamed)
    assert len(model.channels) == len(renamed_model.channels)
    assert model.source_revision != renamed_model.source_revision
    assert canonical_facts([fact("not_isHot", "apple1")]) == frozenset()
    with pytest.raises(ValueError, match="contradictory"):
        canonical_facts([fact("not_isHot", "apple1"), fact("isHot", "apple1")])
    values = model.reachable_values(model.initial, frozenset(), 0)
    assert all(len(v) == 1 for v in values.values())


def test_unknown_token_decoders_retain_broad_current_action_envelope():
    class UnstableDecoder:
        def decode(self, *args, **kwargs):
            raise AssertionError("unknown decoders cannot justify prefix pruning")

    current = session()
    registry = current.record["prefix_registry"]
    schema = resolve_effect_schema(registry, [1, 2, 3], UnstableDecoder())
    assert not schema["prefix_decoder_supported"]
    assert len(schema["spans"]) == 1
    cert = schema["spans"][0]["certificate"]
    assert cert["prefix_length"] == 0
    assert set(cert["immediate_factor_ids"]) == {key for row in registry["actions"] for key in row["changed_factor_ids"]}
