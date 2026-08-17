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
    Expected format:
        <think>some reasoning...</think><action>up/down/left/right/still</action>
    Sokoban action mappings:
    - 0: Still (Invalid Action)
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
        matches = re.findall(r"<action>\s*(up|down|left|right)\s*</action>", actions[i])
        if len(matches) != 1:
            actions[i] = 0
            continue
        actions[i] = action_pools[matches[0]]
        valids[i] = 1

    return actions, valids
