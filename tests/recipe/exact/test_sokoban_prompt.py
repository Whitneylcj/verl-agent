from agent_system.environments.prompts.sokoban import (
    SOKOBAN_TEMPLATE,
    SOKOBAN_TEMPLATE_NO_HIS,
    SOKOBAN_VISUAL_TEMPLATE,
)


def test_sokoban_prompts_require_bounded_spatial_plan_and_action_tag():
    for template in (SOKOBAN_TEMPLATE, SOKOBAN_TEMPLATE_NO_HIS, SOKOBAN_VISUAL_TEMPLATE):
        assert "Number rows top-to-bottom and columns left-to-right" in template
        assert "reply in exactly two lines" in template
        assert "Line 1 must be at most 35 words" in template
        assert "required player cell=(row,column)" in template
        assert "Replace every placeholder" in template
        assert "<action>up</action>" in template
        assert "Never use <up>" in template


def test_sokoban_prompts_explain_push_side_planning_and_loop_avoidance():
    for template in (SOKOBAN_TEMPLATE, SOKOBAN_TEMPLATE_NO_HIS, SOKOBAN_VISUAL_TEMPLATE):
        assert "move the player to the side opposite that push direction" in template
        assert "left the board unchanged" in template
        assert "instead of repeating a loop" in template
        assert "push up needs the player at (r+1,c)" in template
        assert "push down needs the player at (r-1,c)" in template
        assert "Push left needs the player at (r,c+1)" in template
        assert "push right needs the player at (r,c-1)" in template
        assert "execute that push now" in template
