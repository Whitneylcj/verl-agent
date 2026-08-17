from agent_system.environments.prompts.sokoban import (
    SOKOBAN_TEMPLATE,
    SOKOBAN_TEMPLATE_NO_HIS,
    SOKOBAN_VISUAL_TEMPLATE,
)


def test_sokoban_prompts_make_reasoning_optional_and_require_action_tag():
    for template in (SOKOBAN_TEMPLATE, SOKOBAN_TEMPLATE_NO_HIS, SOKOBAN_VISUAL_TEMPLATE):
        assert "Reasoning is optional" in template
        assert "within <action> and </action> tags" in template
        assert "only required executable portion" in template
