# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --------------------- Sokoban --------------------- #
SOKOBAN_TEMPLATE_NO_HIS = """
You are an expert agent operating in the Sokoban environment.

# Symbols and Their Meaning
- Walls (`#`): These block movement. You can't move through or push anything into walls.
- Floor (`_`): Open spaces where you can walk and move boxes.
- Targets (`O`): The spots where boxes need to go.
- Boxes (`X`): These are what you need to push onto the targets.
- Player (`P`): That's you! You'll move around the grid to push boxes.
- Box on Target (`√`): A box successfully placed on a target.
- Player on Target (`S`): You standing on a target.

# Your Goal
Your goal is to push all the boxes (`X`) onto the target spots (`O`). Once all boxes are on the targets, you win!

# Rules
You can only push boxes. You can't pull them, so plan ahead to avoid getting stuck.
You can't walk through or push boxes into walls (`#`).
To avoid traps, do not push boxes into corners or against walls where they can't be moved again.

# Planning Checklist
Locate the player, every box, and every target before choosing an action.
Choose the next useful box push, then move the player to the side opposite that push direction.
If the previous action left the board unchanged, choose a different non-blocked action instead of repeating a loop.

# Push Geometry
For a box at (r,c): push up needs the player at (r+1,c) and an open (r-1,c); push down needs the player at (r-1,c) and an open (r+1,c).
Push left needs the player at (r,c+1) and an open (r,c-1); push right needs the player at (r,c-1) and an open (r,c+1).
If the player is already at the required cell, execute that push now. Otherwise choose one legal move toward the required cell without walking through the box.

# Current Step
Your current observation is:
{current_observation}
Your admissible actions are ["up", "down", "left", "right"].

Now it's your turn to make a move (choose ONE action only for the current step).
Read the grid literally and do not invent boxes or targets. Number rows top-to-bottom and columns left-to-right.
Check the push geometry internally, then reply in exactly two lines without listing alternatives or restating the grid.
Line 1 must be at most 35 words: Plan: P=(row,column); X=(row,column); O=(row,column); push=DIRECTION; required player cell=(row,column); move=DIRECTION.
Replace every placeholder with actual coordinates or one of up, down, left, right.
Line 2 must be exactly one of: <action>up</action>, <action>down</action>, <action>left</action>, <action>right</action>. Never use <up>, <down>, <left>, or <right> as tags.
"""

SOKOBAN_TEMPLATE = """
You are an expert agent operating in the Sokoban environment.

# Symbols and Their Meaning
- Walls (`#`): These block movement. You can't move through or push anything into walls.
- Floor (`_`): Open spaces where you can walk and move boxes.
- Targets (`O`): The spots where boxes need to go.
- Boxes (`X`): These are what you need to push onto the targets.
- Player (`P`): That's you! You'll move around the grid to push boxes.
- Box on Target (`√`): A box successfully placed on a target.
- Player on Target (`S`): You standing on a target.

# Your Goal
Your goal is to push all the boxes (`X`) onto the target spots (`O`). Once all boxes are on the targets, you win!

# Rules
You can only push boxes. You can't pull them, so plan ahead to avoid getting stuck.
You can't walk through or push boxes into walls (`#`).
To avoid traps, do not push boxes into corners or against walls where they can't be moved again.

# Planning Checklist
Locate the player, every box, and every target before choosing an action.
Choose the next useful box push, then move the player to the side opposite that push direction.
If the previous action left the board unchanged, choose a different non-blocked action instead of repeating a loop.

# Push Geometry
For a box at (r,c): push up needs the player at (r+1,c) and an open (r-1,c); push down needs the player at (r-1,c) and an open (r+1,c).
Push left needs the player at (r,c+1) and an open (r,c-1); push right needs the player at (r,c-1) and an open (r,c+1).
If the player is already at the required cell, execute that push now. Otherwise choose one legal move toward the required cell without walking through the box.

# Current Step
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is:
{current_observation}
{loop_warning}
Your admissible actions are ["up", "down", "left", "right"].

Now it's your turn to make a move (choose ONE action only for the current step).
Read the grid literally and do not invent boxes or targets. Number rows top-to-bottom and columns left-to-right.
Check the push geometry internally, then reply in exactly two lines without listing alternatives or restating the grid.
Line 1 must be at most 35 words: Plan: P=(row,column); X=(row,column); O=(row,column); push=DIRECTION; required player cell=(row,column); move=DIRECTION.
Replace every placeholder with actual coordinates or one of up, down, left, right.
Line 2 must be exactly one of: <action>up</action>, <action>down</action>, <action>left</action>, <action>right</action>. Never use <up>, <down>, <left>, or <right> as tags.
"""

SOKOBAN_VISUAL_TEMPLATE = """
You are an expert agent operating in the Sokoban environment. Your goal is to push all the boxes onto the target spots. Once all boxes are on the targets, you win!

# Rules
You can only push boxes. You can't pull them, so plan ahead to avoid getting stuck.
You can't walk through or push boxes into walls.
To avoid traps, do not push boxes into corners or against walls where they can't be moved again.

# Planning Checklist
Locate the player, every box, and every target before choosing an action.
Choose the next useful box push, then move the player to the side opposite that push direction.
If the previous action left the board unchanged, choose a different non-blocked action instead of repeating a loop.

# Push Geometry
For a box at (r,c): push up needs the player at (r+1,c) and an open (r-1,c); push down needs the player at (r-1,c) and an open (r+1,c).
Push left needs the player at (r,c+1) and an open (r,c-1); push right needs the player at (r,c-1) and an open (r,c+1).
If the player is already at the required cell, execute that push now. Otherwise choose one legal move toward the required cell without walking through the box.

# Visual Elements in the Image:
Character: A small, green alien-like figure with two antennae and black eyes. It represents you.
Box: A yellow crate marked with an orange "X" across its front. It is the box you need to push.
Target: A black tile outlined in red, with a small red diamond shape in the center. It marks the destination where a box should be pushed.

# Current Step
Your current observation is shown in the image: <image>
Your admissible actions are ["up", "down", "left", "right"].

Now it's your turn to make a move (choose ONE action only for the current step).
Read the image literally and do not invent boxes or targets. Number rows top-to-bottom and columns left-to-right.
Check the push geometry internally, then reply in exactly two lines without listing alternatives or restating the grid.
Line 1 must be at most 35 words: Plan: P=(row,column); X=(row,column); O=(row,column); push=DIRECTION; required player cell=(row,column); move=DIRECTION.
Replace every placeholder with actual coordinates or one of up, down, left, right.
Line 2 must be exactly one of: <action>up</action>, <action>down</action>, <action>left</action>, <action>right</action>. Never use <up>, <down>, <left>, or <right> as tags.
"""
