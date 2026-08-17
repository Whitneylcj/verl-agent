"""Build the persistent 1k-product Lucene index used by WebShop pilots."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


def product_document(product: Mapping[str, Any]) -> dict[str, Any]:
    bullets = list(product.get("BulletPoints") or ())
    option_texts = []
    for option_name, option_values in dict(product.get("options") or {}).items():
        option_texts.append(f"{option_name}: {', '.join(option_values)}")
    contents = " ".join(
        (
            str(product["Title"]),
            str(product["Description"]),
            str(bullets[0] if bullets else ""),
            ", and ".join(option_texts),
        )
    ).lower()
    return {"id": str(product["asin"]), "contents": contents, "product": dict(product)}


def write_documents(products: Sequence[Mapping[str, Any]], destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    documents = [product_document(product) for product in products[:1000]]
    with destination.open("w", encoding="utf-8") as handle:
        for document in documents:
            json.dump(document, handle, ensure_ascii=False)
            handle.write("\n")
    return len(documents)


def validate_source_files(data_root: Path) -> tuple[int, int, int]:
    """Validate the 1k WebShop product/attribute pair before indexing."""

    product_path = data_root / "items_shuffle_1000.json"
    attribute_path = data_root / "items_ins_v2_1000.json"
    human_instruction_path = data_root / "items_human_ins.json"
    with product_path.open(encoding="utf-8") as handle:
        products = json.load(handle)
    with attribute_path.open(encoding="utf-8") as handle:
        attributes = json.load(handle)
    with human_instruction_path.open(encoding="utf-8") as handle:
        human_instructions = json.load(handle)
    if not isinstance(products, list) or len(products) != 1000:
        raise ValueError("WebShop small product data must contain exactly 1000 products")
    if not isinstance(attributes, Mapping) or len(attributes) != 1000:
        raise ValueError("WebShop small attribute data must contain exactly 1000 entries")
    asins = [str(product["asin"]) for product in products]
    if len(set(asins)) != len(asins):
        raise ValueError("WebShop small product data contains duplicate ASINs")
    missing_attributes = set(asins) - set(attributes)
    if missing_attributes:
        raise ValueError(f"WebShop attributes are missing {len(missing_attributes)} product ASINs")
    if not isinstance(human_instructions, Mapping) or not human_instructions:
        raise ValueError("WebShop human instruction data must be a non-empty mapping")
    human_overlap = set(asins) & set(human_instructions)
    if not human_overlap:
        raise ValueError("WebShop small products have no matching human instructions")
    return len(products), len(attributes), len(human_overlap)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--search-root", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    search_root = args.search_root.expanduser().resolve()
    for name in (
        "items_shuffle_1000.json",
        "items_ins_v2_1000.json",
        "items_human_ins.json",
    ):
        if not (data_root / name).is_file():
            raise FileNotFoundError(f"missing WebShop input: {data_root / name}")
    validate_source_files(data_root)

    resource_dir = search_root / "resources_1k"
    index_dir = search_root / "indexes_1k"
    if index_dir.exists():
        if not args.force:
            raise FileExistsError(f"WebShop index already exists: {index_dir}; pass --force to rebuild")
        shutil.rmtree(index_dir)

    os.environ["WEBSHOP_DATA_ROOT"] = str(data_root)
    os.environ["WEBSHOP_SEARCH_ROOT"] = str(search_root)
    package_root = Path(__file__).resolve().parents[2] / ("agent_system/environments/env_package/webshop/webshop")
    sys.path.insert(0, str(package_root))
    from web_agent_site.engine.engine import load_products
    from web_agent_site.utils import DEFAULT_ATTR_PATH, DEFAULT_FILE_PATH

    products, *_ = load_products(
        filepath=DEFAULT_FILE_PATH,
        attrpath=DEFAULT_ATTR_PATH,
    )
    count = write_documents(products, resource_dir / "documents.jsonl")
    if count == 0:
        raise RuntimeError("WebShop resource builder produced no documents")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyserini.index.lucene",
            "--collection",
            "JsonCollection",
            "--input",
            str(resource_dir),
            "--index",
            str(index_dir),
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--threads",
            "1",
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw",
        ],
        check=True,
    )
    print(f"indexed {count} WebShop products in {index_dir}")


if __name__ == "__main__":
    main()
