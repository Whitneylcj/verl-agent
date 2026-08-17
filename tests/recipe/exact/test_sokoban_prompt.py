from agent_system.environments.prompts.sokoban import (
    SOKOBAN_TEMPLATE,
    SOKOBAN_TEMPLATE_NO_HIS,
    SOKOBAN_VISUAL_TEMPLATE,
)


def test_sokoban_prompts_bound_reasoning_and_require_action_tag():
    for template in (SOKOBAN_TEMPLATE, SOKOBAN_TEMPLATE_NO_HIS, SOKOBAN_VISUAL_TEMPLATE):
        assert "at most two short reasoning sentences" in template
        assert "within <think> and </think> tags" in template
        assert "within <action> and </action> tags" in template
        assert "Do not write anything after </action>" in template
