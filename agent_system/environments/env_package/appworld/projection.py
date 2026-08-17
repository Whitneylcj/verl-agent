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

from typing import List

from recipe.exact.appworld_adapter import compile_json_api_action


def appworld_projection(actions: List[str], action_mode: str = "python"):
    """
    An function to process the actions
    actions: the list of actions to be processeed, it is a list of strings.
    """
    if action_mode not in {"python", "json_api"}:
        raise ValueError(f"unsupported AppWorld action mode: {action_mode}")
    projected_actions = list(actions)
    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string

        if action_mode == "json_api":
            try:
                projected_actions[i] = compile_json_api_action(original_str)
                valids[i] = 1
            except (TypeError, ValueError):
                projected_actions[i] = "raise ValueError('invalid structured action')"
            continue

        # Legacy unrestricted Python mode.
        start_tag = "<code>"
        end_tag = "</code>"
        start_idx = projected_actions[i].find(start_tag)
        end_idx = projected_actions[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <code>...</code> block, mark as invalid
                extracted_action = projected_actions[i][-100:]
                valids[i] = 0
                projected_actions[i] = extracted_action
                continue

            # Extract just the content between the tags
            extracted_action = projected_actions[i][start_idx + len(start_tag) : end_idx]

            projected_actions[i] = extracted_action
            valids[i] = 1

        except Exception:
            extracted_action = projected_actions[i][-100:]
            valids[i] = 0
            projected_actions[i] = extracted_action

        # check <think>...</think>
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        if think_start_idx == -1 or think_end_idx == -1:
            valids[i] = 0

    return projected_actions, valids
