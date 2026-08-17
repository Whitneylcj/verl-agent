"""Resolve WebShop data and index settings without importing its runtime stack."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def resolve_webshop_env_kwargs(*, use_small: bool, human_goals: bool) -> dict[str, Any]:
    """Return aligned product paths and index size for one WebShop runtime."""

    package_data_root = Path(__file__).resolve().parent / "environments/env_package/webshop/webshop/data"
    data_root = Path(os.environ.get("WEBSHOP_DATA_ROOT", package_data_root)).expanduser()
    suffix = "_1000" if use_small else ""
    return {
        "observation_mode": "text",
        "num_products": 1000 if use_small else None,
        "human_goals": human_goals,
        "file_path": str(data_root / f"items_shuffle{suffix}.json"),
        "attr_path": str(data_root / f"items_ins_v2{suffix}.json"),
    }
