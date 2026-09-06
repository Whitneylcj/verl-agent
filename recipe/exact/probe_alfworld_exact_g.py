"""Remote Linux only: real native semantics, commits and credit pipeline.

No LM or training is involved. Planner actions are used only by this test
driver. Character-weighted sparsity here is diagnostic, not an LM result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from recipe.exact.alfworld_adapter import CodepointTokenizer as CharacterTokenizer

TASKS = ("pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place")


def select_games(root: Path, per_type: int):
    selected = {task: [] for task in TASKS}
    for path in sorted(root.rglob("game.tw-pddl")):
        trajectory = path.parent / "traj_data.json"
        if not trajectory.is_file():
            continue
        metadata = json.loads(trajectory.read_text())
        task = metadata.get("task_type")
        if metadata.get("pddl_params", {}).get("object_sliced", False):
            continue
        if task in selected and len(selected[task]) < per_type:
            selected[task].append(path)
        if all(len(paths) == per_type for paths in selected.values()):
            break
    missing = {key: len(paths) for key, paths in selected.items() if len(paths) != per_type}
    if missing:
        raise ValueError(f"need {per_type} distinct supported games per task type: {missing}")
    return selected


def score_records(records, episode_return):
    import numpy as np
    import torch

    from recipe.exact.advantage import compute_exact_advantage

    width = max(len(text) for text, _, _, _ in records)
    masks = torch.tensor([[1] * len(text) + [0] * (width - len(text)) for text, _, _, _ in records])
    count = len(records)
    fields = {
        "traj_uid": ["probe"] * count,
        "exact_step_id": list(range(1, count + 1)),
        "episode_rewards": [episode_return] * count,
        "exact_factor_pre": [pre for _, pre, _, _ in records],
        "exact_factor_post": [post for _, _, post, _ in records],
        "exact_effect_schema": [schema for _, _, _, schema in records],
    }
    batch = SimpleNamespace(batch={"responses": masks, "response_mask": masks.clone(), "attention_mask": masks.clone()}, non_tensor_batch={k: np.array(v, dtype=object) for k, v in fields.items()}, meta_info={})
    _, metrics, traces = compute_exact_advantage(batch, {"mode": "graph", "conservation_tolerance": 1e-10})
    return {"conservation_error": metrics["exact/conservation_error_max"], "character_weighted_comparison": traces[0]["alfworld_comparison"]["metrics"]}


def probe_game(path, horizon, failure=False, guard=True):
    import textworld

    from agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_tw_env import (
        AlfredDemangler,
        AlfredExactInstrumentation,
    )
    from recipe.exact.alfworld_adapter import COMMIT_PREDICATES, resolve_effect_schema

    class CallCounter(textworld.core.Wrapper):
        calls = 0

        def step(self, command):
            self.calls += 1
            return super().step(command)

    counter = CallCounter()
    instrument = AlfredExactInstrumentation(horizon, guard)
    # The planner is explicitly a test driver; training requests it disabled.
    env = textworld.start(str(path), textworld.EnvInfos(won=True, policy_commands=True), wrappers=[AlfredDemangler(), counter, instrument])
    started = time.perf_counter()
    records, pending = [], []
    rejection_seen = False
    try:
        state = env.reset()
        if state["won"]:
            raise RuntimeError("probe requires a nonterminal initial game")
        for index in range(horizon):
            rows = instrument.session.action_rows(instrument.session.state)
            if pending:
                action = pending.pop(0)
            elif failure and rejection_seen:
                action = None  # Invalid responses exercise normal timeout.
            else:
                plan = state["policy_commands"]
                if not plan:
                    raise RuntimeError("official planner has no solution before probe perturbation")
                row = next((row for row in rows if row["command"] == plan[0]), None)
                if row is None:
                    raise RuntimeError("planner command absent from cached applicable operators")
                action = {"op": row["op"], "args": row["args"], "commit": row["op"] in COMMIT_PREDICATES}
                if failure and row["op"] == "take":
                    # Commit the first picked object back where it was. Selection
                    # uses the public action alone, never hidden target matching.
                    pending = [{"op": "put", "args": row["args"], "commit": True}, {"op": "take", "args": row["args"], "commit": False}]
            text = json.dumps(action, separators=(",", ":")) if action is not None else "invalid"
            pre = instrument.session.record
            schema = resolve_effect_schema(pre["prefix_registry"], list(map(ord, text)), CharacterTokenizer())
            state, _, done = env.step(text)
            if counter.calls != index + 1:
                raise AssertionError("one model response must cause exactly one native call")
            rejection_seen |= state["extra.exact"]["reason"] == "protected_fact"
            records.append((text, pre, instrument.session.record, schema))
            if done:
                break
        won = bool(state["won"])
        if failure and (won or not rejection_seen):
            raise AssertionError("failure probe must exercise a rejected destructive action and timeout")
        if not failure and not won:
            raise AssertionError("official planner failed to finish within the horizon")
        # Absorption must not issue a postterminal environment query.
        calls = counter.calls
        env.step("invalid")
        if counter.calls != calls:
            raise AssertionError("terminal logical extension called native env.step")
        return {
            "task_hash": hashlib.sha256(str(path).encode()).hexdigest(),
            "failure_probe": failure,
            "guard": guard,
            "won": won,
            "steps": len(records),
            "native_calls": calls,
            "seconds": time.perf_counter() - started,
            "source_revision": instrument.session.model.source_revision,
            "channel_count": len(instrument.session.model.channels),
            "protected_rejection_checked": rejection_seen,
            **score_records(records, 10.0 * won),
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-root", type=Path, required=True)
    parser.add_argument("--per-type", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux":
        raise RuntimeError("real ALFWorld probes belong on the prepared Linux host")
    if args.per_type < 2 or args.horizon < 10:
        raise ValueError("probe requires at least two games per type and horizon >= 10")
    selected = select_games(args.games_root, args.per_type)
    results = []
    for task, paths in selected.items():
        for path in paths:
            for failure, guard in ((False, False), (False, True), (True, True)):
                results.append({"task_type": task, **probe_game(path, args.horizon, failure, guard)})
    payload = {"status": "passed", "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "token_weighting": "characters; no model sampled", "gradient_variance": None, "tasks": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": "passed", "episodes": len(results), "output": str(args.output)}))


if __name__ == "__main__":
    main()
