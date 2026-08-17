"""Create a fail-closed, secret-bounded manifest before an agentic pilot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

_SENSITIVE_OVERRIDE = re.compile(r"(?i)(api[_-]?(?:key|token)|access[_-]?token|auth[_-]?token|hf[_-]?token|password|passwd|secret|bearer)")
_PACKAGES = ("torch", "transformers", "vllm", "ray", "numpy", "flash-attn")


def _command(args: Sequence[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        list(args),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def redact_override(value: str) -> str:
    """Keep config provenance without persisting obvious credentials."""

    key = value.split("=", 1)[0]
    return f"{key}=[REDACTED]" if _SENSITIVE_OVERRIDE.search(key) else value


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for package in _PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _gpu_inventory() -> list[dict[str, str]]:
    try:
        output = _command(
            (
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            )
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    fields = ("index", "name", "uuid", "memory_total_mib", "driver_version")
    return [dict(zip(fields, (piece.strip() for piece in line.split(",")))) for line in output.splitlines()]


def build_manifest(
    *,
    repo_root: Path,
    experiment: Mapping[str, Any],
    overrides: Sequence[str],
    include_hardware: bool = True,
) -> dict[str, Any]:
    """Build immutable launch evidence from the current checkout and runtime."""

    repo_root = repo_root.expanduser().resolve()
    tracked_status = _command(("git", "status", "--porcelain", "--untracked-files=no"), cwd=repo_root)
    redacted_overrides = [redact_override(value) for value in overrides]
    override_digest = hashlib.sha256(json.dumps(redacted_overrides, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    manifest = {
        "schema_version": "exact.run_manifest.v1",
        "created_unix": time.time(),
        "experiment": dict(experiment),
        "git": {
            "commit": _command(("git", "rev-parse", "HEAD"), cwd=repo_root),
            "branch": _command(("git", "branch", "--show-current"), cwd=repo_root),
            "tracked_status": tracked_status.splitlines(),
            "tracked_dirty": bool(tracked_status),
        },
        "runtime": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "packages": _package_versions(),
            "gpus": _gpu_inventory() if include_hardware else [],
        },
        "paths": {
            key: os.environ.get(key)
            for key in (
                "VERL_ROOT",
                "VERL_AGENT_DATA_ROOT",
                "VERL_AGENT_OUTPUT_ROOT",
                "HF_HOME",
                "ALFWORLD_DATA",
                "APPWORLD_ROOT",
                "WEBSHOP_DATA_ROOT",
                "WEBSHOP_SEARCH_ROOT",
            )
            if os.environ.get(key)
        },
        "hydra_overrides": redacted_overrides,
        "hydra_overrides_sha256": override_digest,
    }
    return manifest


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_manifest(
    output_dir: Path,
    manifest: Mapping[str, Any],
    *,
    require_clean: bool = True,
    resume: bool = False,
) -> Path:
    """Write a new manifest or verify an explicit same-run resume."""

    output_dir = output_dir.expanduser().resolve()
    manifest_path = output_dir / "run_manifest.json"
    if require_clean and manifest["git"]["tracked_dirty"]:
        raise RuntimeError("refusing to launch from a checkout with tracked changes")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume or not manifest_path.is_file():
            raise FileExistsError(f"run directory is not empty: {output_dir}; choose a new MODEL_TAG or set RESUME_RUN=1")
        with manifest_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        invariant_paths = (
            ("git", "commit"),
            ("experiment", "name"),
            ("experiment", "environment"),
            ("experiment", "algorithm"),
            ("experiment", "model_path"),
            ("experiment", "seed"),
        )
        mismatches = [".".join(path) for path in invariant_paths if existing[path[0]][path[1]] != manifest[path[0]][path[1]]]
        if existing.get("hydra_overrides_sha256") != manifest.get("hydra_overrides_sha256"):
            mismatches.append("hydra_overrides_sha256")
        if mismatches:
            raise RuntimeError(f"resume manifest mismatch: {', '.join(mismatches)}")
        return manifest_path
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(manifest_path, manifest)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--exact-mode", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--loss-agg-mode", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    experiment = {
        "name": args.name,
        "environment": args.environment,
        "algorithm": args.algorithm,
        "exact_mode": args.exact_mode,
        "model_path": args.model_path,
        "seed": args.seed,
        "loss_agg_mode": args.loss_agg_mode,
    }
    manifest = build_manifest(
        repo_root=args.repo_root,
        experiment=experiment,
        overrides=args.override,
    )
    path = write_manifest(
        args.output_dir,
        manifest,
        resume=args.resume,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
