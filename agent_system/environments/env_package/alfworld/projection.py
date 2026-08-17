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

import re
from typing import List

_ACTION_PATTERN = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)


def alfworld_response_syntax_valid(response: str) -> bool:
    """Check response structure independently of environment admissibility."""

    response = str(response)
    lowered = response.lower()
    action_matches = _ACTION_PATTERN.findall(response)
    has_one_action = len(action_matches) == 1 and bool(action_matches[0].strip())
    has_reasoning_block = (
        "<think>" in lowered and "</think>" in lowered
    ) or (
        "<thinking>" in lowered and "</thinking>" in lowered
    )
    has_chinese = re.search(r"[\u4e00-\u9fff]", response) is not None
    return has_one_action and has_reasoning_block and not has_chinese


def alfworld_projection(actions: List[str], action_pools: List[List[str]]):
    """
    An function to process the actions
    actions: the list of actions to be processeed, it is a list of strings.
    action_pools: the list of action pools, each pool is a list of strings.
    """

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_response = str(actions[i])
        match = _ACTION_PATTERN.search(original_response)
        if match is None:
            actions[i] = original_response.lower()[-30:]
            continue
        extracted_action = match.group(1).strip().lower()
        actions[i] = extracted_action
        admissible_actions = {str(action).strip().lower() for action in action_pools[i]}
        valids[i] = int(
            alfworld_response_syntax_valid(original_response)
            and extracted_action in admissible_actions
        )

    return actions, valids
