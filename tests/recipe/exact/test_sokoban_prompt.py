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


def test_sokoban_prompts_explain_push_side_planning_and_loop_avoidance():
    for template in (SOKOBAN_TEMPLATE, SOKOBAN_TEMPLATE_NO_HIS, SOKOBAN_VISUAL_TEMPLATE):
        assert "move the player to the side opposite that push direction" in template
        assert "left the board unchanged" in template
        assert "instead of repeating a loop" in template
