from agent_system.environments.prompts.sokoban import (
    SOKOBAN_TEMPLATE,
    SOKOBAN_TEMPLATE_NO_HIS,
    SOKOBAN_VISUAL_TEMPLATE,
)


def test_sokoban_prompts_require_brief_spatial_reasoning_and_action_tag():
    for template in (SOKOBAN_TEMPLATE, SOKOBAN_TEMPLATE_NO_HIS, SOKOBAN_VISUAL_TEMPLATE):
        assert "Number rows top-to-bottom and columns left-to-right" in template
        assert "Before the action, write 1-3 short sentences" in template
        assert "intended useful box push" in template
        assert "within <action> and </action> tags" in template
        assert "only required executable portion" in template


def test_sokoban_prompts_explain_push_side_planning_and_loop_avoidance():
    for template in (SOKOBAN_TEMPLATE, SOKOBAN_TEMPLATE_NO_HIS, SOKOBAN_VISUAL_TEMPLATE):
        assert "move the player to the side opposite that push direction" in template
        assert "left the board unchanged" in template
        assert "instead of repeating a loop" in template
