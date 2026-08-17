"""Download a pinned Hugging Face model snapshot and validate local assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DEFAULT_WEIGHT_SHA256 = {
    "model.safetensors": "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee",
}


def discover_weight_files(snapshot: Path) -> list[Path]:
    """Resolve all safetensor shards required by one local snapshot."""

    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ValueError("model.safetensors.index.json has no weight_map")
        paths = [snapshot / name for name in sorted(set(weight_map.values()))]
    else:
        paths = sorted(snapshot.glob("*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"no safetensors weights found in {snapshot}")
    missing = [path.name for path in paths if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError(f"missing or empty safetensors shards: {missing}")
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_weight_hashes(
    weight_paths: list[Path],
    expected_sha256: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Hash every weight and reject missing, extra, or mismatched expectations."""

    expected_sha256 = {str(name): str(value).lower() for name, value in expected_sha256.items()}
    actual_names = {path.name for path in weight_paths}
    if expected_sha256 and set(expected_sha256) != actual_names:
        raise ValueError(f"expected weight hashes must exactly cover the snapshot weights; expected={sorted(expected_sha256)}, actual={sorted(actual_names)}")
    records = []
    for path in weight_paths:
        sha256 = _sha256(path)
        expected = expected_sha256.get(path.name)
        if expected is not None and sha256 != expected:
            raise ValueError(f"weight SHA-256 mismatch for {path.name}: expected {expected}, got {sha256}")
        records.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256})
    return records


def validate_snapshot(
    snapshot: Path,
    expected_weight_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate weights, config, and tokenizer strictly from local files."""

    snapshot = snapshot.expanduser().resolve()
    required = (snapshot / "config.json", snapshot / "tokenizer_config.json")
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"model snapshot is missing required files: {missing}")
    weight_paths = discover_weight_files(snapshot)

    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    weight_records = validate_weight_hashes(weight_paths, expected_weight_sha256 or {})
    return {
        "resolved_snapshot": str(snapshot),
        "snapshot_commit": snapshot.name if len(snapshot.name) == 40 else None,
        "model_type": getattr(config, "model_type", None),
        "architectures": list(getattr(config, "architectures", ()) or ()),
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_present": bool(getattr(tokenizer, "chat_template", None)),
        "weight_files": weight_records,
        "weight_bytes": sum(path.stat().st_size for path in weight_paths),
    }


def _expected_hashes(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("expected weight hashes must use NAME=SHA256")
        name, sha256 = value.split("=", 1)
        if not name or len(sha256) != 64 or any(character not in "0123456789abcdefABCDEF" for character in sha256):
            raise ValueError(f"invalid expected weight hash: {value}")
        result[name] = sha256.lower()
    return result


def download_snapshot(
    *,
    model: str,
    source: str,
    source_revision: str,
    cache_dir: Path,
) -> Path:
    if source == "huggingface":
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                repo_id=model,
                revision=source_revision,
                cache_dir=cache_dir,
            )
        )
    if source == "modelscope":
        from modelscope import snapshot_download

        return Path(
            snapshot_download(
                model_id=model,
                revision=source_revision,
                cache_dir=cache_dir,
                max_workers=8,
            )
        )
    raise ValueError(f"unsupported model source: {source}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--source", choices=("huggingface", "modelscope"), default="huggingface")
    parser.add_argument("--source-revision")
    parser.add_argument("--expected-weight-sha256", action="append", default=[])
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    args = parser.parse_args()

    source_revision = args.source_revision or (args.revision if args.source == "huggingface" else "master")
    expected_hashes = _expected_hashes(args.expected_weight_sha256)
    if not expected_hashes and args.model == DEFAULT_MODEL and args.revision == DEFAULT_REVISION:
        expected_hashes = dict(DEFAULT_WEIGHT_SHA256)
    if args.source == "modelscope" and not expected_hashes:
        raise ValueError("ModelScope downloads require explicit expected weight SHA-256 values")
    snapshot = download_snapshot(
        model=args.model,
        source=args.source,
        source_revision=source_revision,
        cache_dir=args.cache_dir.expanduser().resolve(),
    )
    validation = validate_snapshot(snapshot, expected_weight_sha256=expected_hashes)
    if args.source == "huggingface" and validation["snapshot_commit"] != args.revision:
        raise RuntimeError(f"resolved snapshot {validation['snapshot_commit']} does not match pinned revision {args.revision}")
    manifest = {
        "schema_version": "exact.model_manifest.v1",
        "created_unix": time.time(),
        "repo_id": args.model,
        "requested_revision": args.revision,
        "download_source": args.source,
        "source_revision": source_revision,
        **validation,
    }
    _atomic_json(args.output_manifest.expanduser().resolve(), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
