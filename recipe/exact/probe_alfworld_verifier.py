"""Run the EXACT verifier against one official ALFWorld game per text task.

This is a remote-only integration probe. It follows ALFWorld's hand-coded
expert without printing benchmark instance paths, observations, or actions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import (
    get_environment,
)
from agent_system.environments.env_package.alfworld.envs import (
    AlfworldWorker,
    load_config_file,
)
from recipe.exact.alfworld_verifier import (
    SUPPORTED_ALFWORLD_TASKS,
    load_alfworld_verifier_spec,
)
from recipe.exact.core_exact import IdentityPotential, build_scoped_conserved_atoms
from recipe.exact.credit_spec import FactorSnapshot


def _single(value: Any, name: str) -> Any:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__") or len(value) != 1:
        raise RuntimeError(f"expected exactly one batched ALFWorld {name} value")
    return value[0]


def _select_one_game_per_task(game_files: list[str]) -> dict[str, str]:
    selected = {}
    for gamefile in game_files:
        spec = load_alfworld_verifier_spec(gamefile)
        selected.setdefault(spec.task_type, gamefile)
    missing = set(SUPPORTED_ALFWORLD_TASKS) - set(selected)
    if missing:
        raise RuntimeError(f"ALFWorld data is missing verifier task types: {sorted(missing)}")
    return selected


def _compile_real_trajectory(
    snapshots: list[dict[str, Any]],
    episode_return: float,
) -> float:
    typed_snapshots = tuple(
        FactorSnapshot(
            checkpoint_id=checkpoint_id,
            factor_ids=tuple(snapshot["factor_ids"]),
            values=np.asarray(snapshot["values"], dtype=np.float64),
            read_sets=snapshot["read_sets"],
            schema_version=snapshot["schema_version"],
            channel_roles=snapshot["channel_roles"],
            potential_weights=np.asarray(
                snapshot["potential_weights"],
                dtype=np.float64,
            ),
            source_revision=snapshot["source_revision"],
        )
        for checkpoint_id, snapshot in enumerate(snapshots)
    )
    potential = IdentityPotential(
        factor_ids=typed_snapshots[0].factor_ids,
        weights=typed_snapshots[0].potential_weights,
    )
    conserved = build_scoped_conserved_atoms(
        typed_snapshots,
        episode_return,
        potential,
    )
    if conserved.channel_target_masses["terminal_success"] != episode_return:
        raise RuntimeError("ALFWorld terminal verifier does not reconstruct episode return")
    process_target_mass = sum(
        value
        for factor_id, value in conserved.channel_target_masses.items()
        if factor_id != "terminal_success"
    )
    if process_target_mass != 0.0 or conserved.opaque_target_atom.value != 0.0:
        raise RuntimeError("ALFWorld process verifier changed the conserved return target")
    return conserved.conservation_error


def _probe_game(base_env: Any, gamefile: str, max_steps: int) -> dict[str, Any]:
    verifier_spec = load_alfworld_verifier_spec(gamefile)
    base_env.game_files = [gamefile]
    base_env.num_games = 1
    base_env.use_expert = True
    worker = AlfworldWorker({}, seed=0, base_env=base_env, deterministic_reset=True)
    try:
        _, infos = worker.reset()
        snapshots = [worker.exact_credit_snapshot()]
        factor_ids = snapshots[0]["factor_ids"]
        total_reward = 0.0
        done = False

        for step_count in range(1, max_steps + 1):
            plan = _single(infos["extra.expert_plan"], "expert plan")
            if not plan:
                raise RuntimeError("ALFWorld expert plan ended before official success")
            action = plan[0]
            admissible = _single(infos["admissible_commands"], "admissible commands")
            if action not in admissible:
                raise RuntimeError("ALFWorld expert proposed a non-admissible action")

            _, _, dones, infos = worker.step(action)
            won = bool(_single(infos["won"], "won"))
            done = bool(_single(dones, "done"))
            total_reward += 10.0 * float(won)
            snapshot = worker.exact_credit_snapshot()
            if snapshot["factor_ids"] != factor_ids:
                raise RuntimeError("ALFWorld factor schema changed during expert rollout")
            snapshots.append(snapshot)
            if done:
                break
        else:
            step_count = max_steps

        if not done or not bool(_single(infos["won"], "won")):
            raise RuntimeError("ALFWorld expert did not reach official success")
        if total_reward != 10.0:
            raise RuntimeError("ALFWorld training return no longer equals 10 * won")
        if snapshots[-1]["values"][0] != 1.0:
            raise RuntimeError("ALFWorld terminal return component did not activate")

        process_values = [snapshot["values"][1:] for snapshot in snapshots]
        process_change_count = sum(
            current != previous
            for previous, current in zip(process_values, process_values[1:])
        )
        if process_change_count == 0:
            raise RuntimeError("ALFWorld process verifier never changed on an expert plan")
        if len(factor_ids) > 2 and not any(any(values) for values in process_values[1:-1]):
            raise RuntimeError("multi-condition ALFWorld verifier exposed no preterminal progress")
        conservation_error = _compile_real_trajectory(snapshots, total_reward)

        return {
            "task_type": verifier_spec.task_type,
            "steps": step_count,
            "factor_ids": factor_ids,
            "process_change_count": process_change_count,
            "preterminal_progress": any(any(values) for values in process_values[1:-1]),
            "episode_return": total_reward,
            "conservation_error": conservation_error,
        }
    finally:
        worker.env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("agent_system/environments/env_package/alfworld/configs/config_tw.yaml"),
    )
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    config = load_config_file(args.config)
    config["env"]["domain_randomization"] = False
    env_type = config["env"]["type"]
    base_env = get_environment(env_type)(config, train_eval="train")
    selected = _select_one_game_per_task(base_env.game_files)
    results = [
        _probe_game(base_env, selected[task_type], args.max_steps)
        for task_type in SUPPORTED_ALFWORLD_TASKS
    ]
    print(json.dumps({"status": "passed", "tasks": results}, indent=2))


if __name__ == "__main__":
    main()
