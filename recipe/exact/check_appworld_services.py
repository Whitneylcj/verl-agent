"""Fail fast when an AppWorld pilot has too few reachable service instances."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from typing import Sequence


def load_port_manifest(path: str | Path) -> list[int]:
    manifest = Path(path).expanduser()
    if not manifest.is_file():
        raise FileNotFoundError(f"AppWorld port manifest does not exist: {manifest}")
    ports = []
    for line_number, raw_line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        value = raw_line.strip()
        if not value:
            continue
        if not value.isdigit() or not 1 <= int(value) <= 65_535:
            raise ValueError(f"invalid AppWorld port at {manifest}:{line_number}: {value!r}")
        ports.append(int(value))
    if len(ports) != len(set(ports)):
        raise ValueError(f"AppWorld port manifest contains duplicates: {manifest}")
    return ports


def require_reachable_services(
    port_file: str | Path,
    required: int,
    host: str = "127.0.0.1",
    timeout: float = 1.0,
) -> list[int]:
    if required <= 0:
        raise ValueError("required AppWorld service count must be positive")
    ports = load_port_manifest(port_file)
    if len(ports) < required:
        raise RuntimeError(f"AppWorld needs {required} services, but the manifest contains only {len(ports)} ports")
    selected = ports[:required]
    unreachable = []
    for port in selected:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                pass
        except OSError:
            unreachable.append(port)
    if unreachable:
        raise RuntimeError(f"AppWorld services are unreachable at {host}: {unreachable}")
    return selected


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port-file", required=True)
    parser.add_argument("--required", required=True, type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--timeout", default=1.0, type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    ports = require_reachable_services(args.port_file, args.required, args.host, args.timeout)
    print(json.dumps({"host": args.host, "required": args.required, "ports": ports}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
