"""Small PDDL problems using the actual vendored domain and goal library."""

import json
import runpy
from dataclasses import dataclass
from types import SimpleNamespace

from recipe.exact.alfworld_adapter import AlfworldExactSession, CodepointTokenizer
from recipe.exact.alfworld_semantics import DOMAIN_PATH, Fact, truth_values

CharacterTokenizer = CodepointTokenizer

TASKS = (
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
)


def fact(predicate, *args):
    return Fact(predicate.lower(), tuple(arg.lower() for arg in args))


def game_data(task=TASKS[2]):
    goals = runpy.run_path(str(DOMAIN_PATH.parents[1] / "gen/goal_library.py"))["gdict"]
    goal = goals[task]["pddl"].format(obj="Apple", recep="Fridge", toggle="DeskLamp").replace("#", "-")
    objects = """a - agent
      lt ls lm lf lf2 - location
      table sink micro fridge fridge2 - receptacle
      apple1 apple2 lamp knife - object
      AppleType DeskLampType KnifeType ButterKnifeType - otype
      TableType SinkBasinType MicrowaveType FridgeType - rtype"""
    init = ["(atLocation a lt)", "(toggleable lamp)", "(objectType lamp DeskLampType)", "(objectType knife KnifeType)"]
    for receptacle, location, typ in (("table", "lt", "Table"), ("sink", "ls", "SinkBasin"), ("micro", "lm", "Microwave"), ("fridge", "lf", "Fridge"), ("fridge2", "lf2", "Fridge")):
        init += [f"(receptacleAtLocation {receptacle} {location})", f"(receptacleType {receptacle} {typ}Type)"]
    for typ in ("Table", "SinkBasin", "Microwave", "Fridge"):
        init += [f"(canContain {typ}Type AppleType)", f"(canContain {typ}Type KnifeType)"]
    init += ["(openable fridge)", "(openable micro)"]
    for obj in ("apple1", "apple2", "lamp", "knife"):
        init += [f"(inReceptacle {obj} table)", f"(objectAtLocation {obj} lt)"]
    init += ["(pickupable knife)"]
    for obj in ("apple1", "apple2"):
        init += [f"(objectType {obj} AppleType)"]
        init += [f"({prop} {obj})" for prop in ("pickupable", "cleanable", "heatable", "coolable", "sliceable")]
    problem = f"(define (problem test) (:domain alfred) (:objects {objects}) (:init {' '.join(init)}) {goal}"
    return {"pddl_domain": DOMAIN_PATH.read_text(), "pddl_problem": problem}


@dataclass(frozen=True)
class Named:
    name: str


def native_state(model, facts):
    actions, commands = [], []
    for action in model.actions:
        if truth_values(action.precondition, {f: frozenset({True}) for f in facts}) != frozenset({True}):
            continue
        actions.append(SimpleNamespace(name=action.name, mapping={Named(k.lstrip("?")): Named(v) for k, v in action.bindings}, preconditions=[], postconditions=[]))
        commands.append(action.name + " " + " ".join(v for _, v in sorted(action.bindings)))
    return {"_facts": facts, "won": model.goal_holds(facts), "_valid_actions": actions, "_valid_commands": commands, "_entity_infos": {name: Named(name) for name in model.objects}}


def session(task=TASKS[2], horizon=30, guard=True):
    result = AlfworldExactSession(game_data(task), horizon, guard_enabled=guard)
    result.capture(native_state(result.model, result.model.initial))
    return result


def response(op, *args, commit=False):
    return json.dumps({"op": op, "args": args, "commit": commit}, separators=(",", ":"))


def execute(current, op, *args, commit=False):
    decision = current.prepare(response(op, *args, commit=commit))
    after = decision.expected_facts if decision.execution_valid else current.state["_facts"]
    current.finish(decision, native_state(current.model, after))
    return decision
