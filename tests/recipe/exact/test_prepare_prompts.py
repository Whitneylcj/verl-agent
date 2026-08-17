from recipe.exact.prepare_prompts import build_prompt_rows


def test_build_prompt_rows_has_agent_schema_and_exact_size():
    rows = build_prompt_rows(3, "train")

    assert len(rows) == 3
    assert rows[0] == {
        "data_source": "text",
        "prompt": [{"role": "user", "content": ""}],
        "ability": "agent",
        "extra_info": {"split": "train", "index": 0},
    }
    assert rows[-1]["extra_info"] == {"split": "train", "index": 2}
