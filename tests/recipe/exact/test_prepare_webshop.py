import json

from recipe.exact.prepare_webshop import product_document, validate_source_files, write_documents


def _product():
    return {
        "asin": "A-1",
        "Title": "Blue Mug",
        "Description": "Ceramic cup",
        "BulletPoints": ["Dishwasher safe"],
        "options": {"size": ["small", "large"]},
    }


def test_product_document_is_searchable_and_keeps_payload():
    document = product_document(_product())
    assert document["id"] == "A-1"
    assert "blue mug" in document["contents"]
    assert "size: small, large" in document["contents"]
    assert document["product"]["asin"] == "A-1"


def test_write_documents_emits_jsonl(tmp_path):
    destination = tmp_path / "resources_1k" / "documents.jsonl"
    assert write_documents([_product()], destination) == 1
    assert json.loads(destination.read_text())["id"] == "A-1"


def test_validate_source_files_requires_aligned_unique_1k_products(tmp_path):
    products = [{"asin": f"A-{index}"} for index in range(1000)]
    attributes = {product["asin"]: {"attributes": []} for product in products}
    human_instructions = {"A-0": [{"instruction": "find it"}]}
    (tmp_path / "items_shuffle_1000.json").write_text(json.dumps(products))
    (tmp_path / "items_ins_v2_1000.json").write_text(json.dumps(attributes))
    (tmp_path / "items_human_ins.json").write_text(json.dumps(human_instructions))

    assert validate_source_files(tmp_path) == (1000, 1000, 1)

    products[-1]["asin"] = products[0]["asin"]
    (tmp_path / "items_shuffle_1000.json").write_text(json.dumps(products))
    try:
        validate_source_files(tmp_path)
    except ValueError as error:
        assert "duplicate ASINs" in str(error)
    else:
        raise AssertionError("duplicate WebShop products were accepted")
