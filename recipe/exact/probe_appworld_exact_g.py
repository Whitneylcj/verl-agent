"""Run a real AppWorld/Ray Exact-G probe without loading a language model."""

from __future__ import annotations

import argparse
import json
import os
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from omegaconf import OmegaConf

from agent_system.environments.env_manager import AppWorldEnvironmentManager
from agent_system.environments.env_package.appworld import (
    appworld_projection,
    build_appworld_envs,
)


class _CharacterTokenizer:
    def decode(self, token_ids: Sequence[int], **kwargs: Any) -> str:
        if not kwargs.get("skip_special_tokens", False):
            raise ValueError("probe tokenizer expects skip_special_tokens=True")
        return "".join(chr(int(token_id)) for token_id in token_ids)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def run_probe(
    *,
    dataset_name: str = "train",
    port_file: str = "appworld_ports.ports",
) -> dict[str, Any]:
    config = OmegaConf.create(
        {
            "env": {
                "env_name": "AppWorld",
                "history_length": 2,
                "appworld": {"action_mode": "json_api"},
            }
        }
    )
    envs = build_appworld_envs(
        dataset_name=dataset_name,
        max_interactions=2,
        seed=0,
        env_num=1,
        group_n=1,
        start_server_id=0,
        resources_per_worker={"num_cpus": 0.1},
        port_file=port_file,
    )
    manager = AppWorldEnvironmentManager(
        envs,
        partial(appworld_projection, action_mode="json_api"),
        config,
    )
    try:
        _, reset_infos = manager.reset(kwargs=None)
        pre_snapshot = manager.exact_credit_snapshots()[0]
        registry = manager.exact_effect_schemas([pre_snapshot])[0]
        if registry["kind"] != "appworld-prefix-effect-v1":
            raise AssertionError("unexpected AppWorld prefix schema")
        if not registry["version_supported"]:
            raise AssertionError("installed AppWorld version is not graph-audited")
        if registry["factor_schema_compile_fallback"]:
            raise AssertionError(registry["factor_schema_compile_error"])
        if registry["opaque_factor_count"]:
            raise AssertionError("selected task unexpectedly contains opaque factors")

        text_action = '<think>Inspect the available apps without changing task state.</think><action>{"app":"api_docs","api":"show_app_descriptions","arguments":{}}</action>'
        token_ids = np.asarray([[ord(character) for character in text_action]], dtype=np.int64)
        concrete = manager.resolve_exact_effect_schemas(
            [registry],
            [text_action],
            token_ids,
            np.ones_like(token_ids),
            _CharacterTokenizer(),
        )[0]
        if concrete["resolution_fallback"]:
            raise AssertionError("valid structured action did not resolve an argument span")
        span_buckets = [span["bucket"] for span in concrete["spans"]]
        if span_buckets != ["appworld.selector", "appworld.arguments"]:
            raise AssertionError(f"unexpected resolved spans: {span_buckets}")

        _, rewards, dones, step_infos = manager.step([text_action])
        post_snapshot = manager.exact_credit_snapshots()[0]
        if post_snapshot["factor_ids"] != pre_snapshot["factor_ids"]:
            raise AssertionError("AppWorld factor schema changed after a read-only API call")
        changed_factor_count = sum(
            before != after
            for before, after in zip(
                pre_snapshot["values"],
                post_snapshot["values"],
                strict=True,
            )
        )
        result = {
            "schema_version": "exact.appworld.real-probe.v1",
            "status": "pass",
            "task_id": reset_infos[0]["task_id"],
            "factor_count": len(pre_snapshot["factor_ids"]),
            "opaque_factor_count": int(registry["opaque_factor_count"]),
            "api_count": len(registry["api_possible_write_sets"]),
            "resolved_span_buckets": span_buckets,
            "resolution_fallback": bool(concrete["resolution_fallback"]),
            "action_valid": bool(step_infos[0]["is_action_valid"]),
            "reward": float(rewards[0]),
            "done": bool(dones[0]),
            "changed_factor_count": changed_factor_count,
        }
        if not result["action_valid"]:
            raise AssertionError("real AppWorld rejected the strict JSON API action")
        return result
    finally:
        manager.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="train")
    parser.add_argument("--port-file", default=os.environ.get("APPWORLD_PORT_FILE", "appworld_ports.ports"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run_probe(dataset_name=args.dataset, port_file=args.port_file)
    if args.output is not None:
        _write_json_atomic(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
