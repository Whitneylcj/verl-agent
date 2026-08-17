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


def sokoban_projection(actions: List[str]):
    """
    A function to process the actions.
    actions: the list of actions to be processed, it is a list of strings.
    Reasoning text and wrappers may surround the action. Prefer exactly one
    <action>up/down/left/right</action> tag. If that tag is absent, accept only
    consistent explicit ``Action: direction`` or ``<direction></direction>``
    markers; free prose and conflicting markers remain invalid.
    Sokoban action mappings:
    - 0: Still (projection fallback for an invalid action)
    - 1: Up
    - 2: Down
    - 3: Left
    - 4: Right
    """

    action_pools = {
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
        "still": 0,
    }

    valids = [0] * len(actions)

    for i in range(len(actions)):
        actions[i] = actions[i].lower()
        canonical = re.findall(r"<action>\s*(up|down|left|right)\s*</action>", actions[i])
        if len(canonical) > 1:
            actions[i] = 0
            continue
        explicit_lines = re.findall(r"(?m)^\s*action\s*:\s*(up|down|left|right)\s*$", actions[i])
        direction_tags = re.findall(r"<(up|down|left|right)>\s*</\1>", actions[i])
        if not canonical and ("<action" in actions[i] or "</action" in actions[i]):
            actions[i] = 0
            continue
        candidates = [*canonical, *explicit_lines, *direction_tags]
        if not candidates or len(set(candidates)) != 1:
            actions[i] = 0
            continue
        actions[i] = action_pools[candidates[0]]
        valids[i] = 1

    return actions, valids
