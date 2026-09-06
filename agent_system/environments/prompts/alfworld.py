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

ALFWORLD_EXACT_JSON_TEMPLATE = '''Complete the household task using one action per response.
Task: {task}
Recent history: {history}
Current observation: {observation}
Available actions (commit=false examples): {actions}
Your previously committed protections: {protections}
Return only a JSON object with exactly these fields in this order: op, args, commit.
Use the operation names and object names shown in the available actions.
You may set commit=true on heat, cool, clean, toggle, or put to permanently
protect the object property or placement established by that successful action
when commit guards are enabled. Runs without guards ignore the commit flag.
Commit only when you intend to preserve it for the rest of this task. There is
no undo. Moves that destroy a protected fact are rejected and consume one step.
Do not commit navigation, hand occupancy, or container open/closed state.
Example: {{"op":"heat","args":["mug 1","microwave 1"],"commit":true}}
'''

# --------------------- ALFWorld --------------------- #
ALFWORLD_QWEN_SMALL_GUIDED_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
Reply with exactly two non-empty lines and no text outside the tags.
Line 1: <thinking>Briefly justify the next action in at most 35 words.</thinking>
Line 2: <action>copy one current admissible action exactly</action>
Never invent, paraphrase, or combine actions.
"""

ALFWORLD_QWEN_SMALL_GUIDED_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
Reply with exactly two non-empty lines and no text outside the tags.
Line 1: <thinking>Briefly justify the next action in at most 35 words.</thinking>
Line 2: <action>copy one current admissible action exactly</action>
Never invent, paraphrase, or combine actions.
"""

ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""
