"""Download a pinned Hugging Face model snapshot and validate local assets."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


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


def validate_snapshot(snapshot: Path) -> dict[str, Any]:
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
    return {
        "resolved_snapshot": str(snapshot),
        "snapshot_commit": snapshot.name if len(snapshot.name) == 40 else None,
        "model_type": getattr(config, "model_type", None),
        "architectures": list(getattr(config, "architectures", ()) or ()),
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_present": bool(getattr(tokenizer, "chat_template", None)),
        "weight_files": [{"name": path.name, "bytes": path.stat().st_size} for path in weight_paths],
        "weight_bytes": sum(path.stat().st_size for path in weight_paths),
    }


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
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=args.model,
            revision=args.revision,
            cache_dir=args.cache_dir.expanduser().resolve(),
        )
    )
    validation = validate_snapshot(snapshot)
    if validation["snapshot_commit"] != args.revision:
        raise RuntimeError(f"resolved snapshot {validation['snapshot_commit']} does not match pinned revision {args.revision}")
    manifest = {
        "schema_version": "exact.model_manifest.v1",
        "created_unix": time.time(),
        "repo_id": args.model,
        "requested_revision": args.revision,
        **validation,
    }
    _atomic_json(args.output_manifest.expanduser().resolve(), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
