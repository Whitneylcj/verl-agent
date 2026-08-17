import json

from recipe.exact.prepare_webshop import product_document, write_documents


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
