from agent_system.environments.prompts.sokoban import (
    SOKOBAN_TEMPLATE,
    SOKOBAN_TEMPLATE_NO_HIS,
    SOKOBAN_VISUAL_TEMPLATE,
)


def test_sokoban_prompts_require_one_line_reasoning_and_action_contract():
    for template in (SOKOBAN_TEMPLATE, SOKOBAN_TEMPLATE_NO_HIS, SOKOBAN_VISUAL_TEMPLATE):
        assert "exactly this one-line template" in template
        assert "<think>one short sentence</think><action>direction</action>" in template
        assert "one lowercase admissible action" in template
        assert "Do not add text or repeat a tag" in template
