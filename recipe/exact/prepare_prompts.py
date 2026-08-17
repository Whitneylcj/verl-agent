"""Create deterministic empty-prompt parquet inputs for agent environments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


def build_prompt_rows(size: int, split: str) -> list[dict[str, Any]]:
    """Return the modality placeholders consumed by the agent rollout collector."""

    if size <= 0:
        raise ValueError("prompt dataset size must be positive")
    return [
        {
            "data_source": "text",
            "prompt": [{"role": "user", "content": ""}],
            "ability": "agent",
            "extra_info": {"split": split, "index": index},
        }
        for index in range(size)
    ]


def write_prompt_datasets(output_dir: Path, train_size: int, validation_size: int) -> tuple[Path, Path]:
    """Write size-exact parquet files atomically without downloading a source dataset."""

    from datasets import Dataset

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (output_dir / "train.parquet", output_dir / "test.parquet")
    rows_by_path = (
        (paths[0], build_prompt_rows(train_size, "train")),
        (paths[1], build_prompt_rows(validation_size, "test")),
    )
    for path, rows in rows_by_path:
        temporary = path.with_suffix(path.suffix + ".tmp")
        Dataset.from_list(rows).to_parquet(str(temporary))
        os.replace(temporary, path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-size", required=True, type=int)
    parser.add_argument("--validation-size", required=True, type=int)
    args = parser.parse_args()

    train_path, validation_path = write_prompt_datasets(
        args.output_dir,
        train_size=args.train_size,
        validation_size=args.validation_size,
    )
    print(f"train_file={train_path}")
    print(f"validation_file={validation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
