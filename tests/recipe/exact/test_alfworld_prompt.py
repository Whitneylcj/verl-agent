import importlib.util
from pathlib import Path

from recipe.exact.alfworld_verifier import ALFWORLD_TASK_FACTOR_IDS

_PROMPT_PATH = Path(__file__).parents[3] / "agent_system/environments/prompts/alfworld.py"
_SPEC = importlib.util.spec_from_file_location("alfworld_prompt_under_test", _PROMPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
ALFWORLD_TEMPLATE = _MODULE.ALFWORLD_TEMPLATE
ALFWORLD_TEMPLATE_NO_HIS = _MODULE.ALFWORLD_TEMPLATE_NO_HIS


def test_alfworld_prompts_require_a_short_two_line_response():
    for template in (ALFWORLD_TEMPLATE, ALFWORLD_TEMPLATE_NO_HIS):
        assert "exactly two non-empty lines" in template
        assert "at most 35 words" in template
        assert "copy one current admissible action exactly" in template
        assert "Never invent, paraphrase, or combine actions" in template


def test_alfworld_prompts_do_not_expose_training_only_verifier_channels():
    prompt_source = f"{ALFWORLD_TEMPLATE}\n{ALFWORLD_TEMPLATE_NO_HIS}"
    assert "terminal_success" not in prompt_source
    for factor_ids in ALFWORLD_TASK_FACTOR_IDS.values():
        for factor_id in factor_ids:
            assert factor_id not in prompt_source
