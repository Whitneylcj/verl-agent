"""Audit Exact-G factor/effect schemas against installed AppWorld task data."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Sequence

import appworld
from appworld.apps import get_all_apps
from appworld.collections.api_docs import ApiDocCollection
from appworld.common.inspect import get_name_to_model
from appworld.common.path_store import path_store
from appworld.ground_truth import GroundTruth

from recipe.exact.appworld_schema import (
    appworld_factor_id,
    build_appworld_effect_registry,
    compile_appworld_factor_reads,
)


def _task_ids() -> list[str]:
    tasks_root = Path(path_store.data) / "tasks"
    return sorted(path.name for path in tasks_root.iterdir() if not path.name.startswith("_") and (path / "ground_truth" / "evaluation.py").is_file())


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _git_state() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "status",
            "--short",
            "--untracked-files=all",
            "--",
            "recipe/exact",
            "verl",
            "agent_system",
            "examples/exact_trainer",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "commit": commit,
        "tracked_clean": not status,
        "tracked_status": status.splitlines(),
    }


def audit_appworld_schema(
    task_ids: Sequence[str] | None = None,
    max_opaque_rate: float = 0.01,
) -> dict[str, Any]:
    if not 0 <= max_opaque_rate <= 1:
        raise ValueError("max_opaque_rate must be between zero and one")
    app_names = tuple(sorted(set(get_all_apps()) | {"admin"}))
    app_to_model_names = {app_name: tuple(sorted(get_name_to_model(app_name))) for app_name in app_names}
    api_docs = ApiDocCollection.load(load_apps=tuple(get_all_apps(skip_admin=True)))
    registry = build_appworld_effect_registry(
        api_docs=api_docs,
        app_to_model_names=app_to_model_names,
        appworld_version=str(appworld.__version__),
    )
    all_resources = tuple(registry["all_model_resources"])
    complete_task_inventory = task_ids is None
    selected_task_ids = list(task_ids) if task_ids is not None else _task_ids()
    git_state = _git_state()

    task_failures = []
    task_failure_count = 0
    mismatch_samples = []
    opaque_samples = []
    factor_count = 0
    opaque_factor_count = 0
    tasks_with_opaque_factors = 0
    for task_id in selected_task_ids:
        try:
            ground_truth = GroundTruth.load(task_id, mode="minimal")
            compiled = compile_appworld_factor_reads(
                ground_truth.evaluation_code,
                all_resources,
                known_values={
                    "public_data": ground_truth.public_data,
                    "private_data": ground_truth.private_data,
                },
            )
            expected_ids = {appworld_factor_id(str(entry["requirement"])) for entry in (ground_truth.test_data or [])}
            actual_ids = set(compiled.read_sets)
            missing = sorted(expected_ids - actual_ids)
            extra = sorted(actual_ids - expected_ids)
            if missing or extra:
                mismatch_samples.append({"task_id": task_id, "missing_factor_ids": missing, "extra_factor_ids": extra})
            factor_count += len(expected_ids)
            opaque_ids = sorted(set(compiled.opaque_factor_ids) & expected_ids)
            opaque_factor_count += len(opaque_ids)
            if opaque_ids:
                tasks_with_opaque_factors += 1
                if len(opaque_samples) < 20:
                    opaque_samples.append({"task_id": task_id, "opaque_factor_ids": opaque_ids})
        except Exception as error:
            task_failure_count += 1
            if len(task_failures) < 20:
                task_failures.append({"task_id": task_id, "error": f"{type(error).__name__}: {error}"})

    opaque_rate = opaque_factor_count / max(factor_count, 1)
    passed = task_failure_count == 0 and not mismatch_samples and bool(registry["version_supported"]) and opaque_rate <= max_opaque_rate and git_state["tracked_clean"]
    return {
        "schema_version": "exact.appworld.schema-audit.v1",
        "passed": passed,
        "appworld_version": str(appworld.__version__),
        "version_supported": bool(registry["version_supported"]),
        "git": git_state,
        "complete_task_inventory": complete_task_inventory,
        "task_count": len(selected_task_ids),
        "factor_count": factor_count,
        "opaque_factor_count": opaque_factor_count,
        "opaque_factor_rate": opaque_rate,
        "max_opaque_rate": max_opaque_rate,
        "tasks_with_opaque_factors": tasks_with_opaque_factors,
        "model_resource_count": len(all_resources),
        "api_count": len(registry["api_possible_write_sets"]),
        "task_failure_count": task_failure_count,
        "factor_mismatch_count": len(mismatch_samples),
        "task_failure_samples": task_failures,
        "factor_mismatch_samples": mismatch_samples[:20],
        "opaque_factor_samples": opaque_samples,
        "effect_certificate": registry["certificate"],
    }


def verify_appworld_schema_audit(path: str | Path) -> dict[str, Any]:
    audit_path = Path(path).expanduser().resolve()
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    current_git = _git_state()
    errors = []
    if report.get("schema_version") != "exact.appworld.schema-audit.v1":
        errors.append("unsupported schema_version")
    if not report.get("passed", False):
        errors.append("AppWorld schema audit did not pass")
    if not report.get("complete_task_inventory", False):
        errors.append("AppWorld schema audit did not cover the complete task inventory")
    if report.get("git", {}).get("commit") != current_git["commit"]:
        errors.append("AppWorld schema audit commit does not match the current checkout")
    if not report.get("git", {}).get("tracked_clean", False):
        errors.append("AppWorld schema audit was generated from a tracked-dirty checkout")
    if not current_git["tracked_clean"]:
        errors.append("current checkout has tracked changes")
    if report.get("appworld_version") != str(appworld.__version__):
        errors.append("AppWorld schema audit package version does not match the runtime")
    return {
        "status": "pass" if not errors else "no_go",
        "audit_path": str(audit_path),
        "commit": current_git["commit"],
        "appworld_version": str(appworld.__version__),
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--max-opaque-rate", type=float, default=0.01)
    args = parser.parse_args(argv)
    if args.verify is not None:
        result = verify_appworld_schema_audit(args.verify)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["status"] == "pass" else 1

    task_ids = None
    if args.max_tasks is not None:
        if args.max_tasks <= 0:
            parser.error("--max-tasks must be positive")
        task_ids = _task_ids()[: args.max_tasks]
    result = audit_appworld_schema(task_ids, max_opaque_rate=args.max_opaque_rate)
    _atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
