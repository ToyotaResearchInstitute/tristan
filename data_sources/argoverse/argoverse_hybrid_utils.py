"""
Variable definitions for hybrid prediction in Argoverse.
"""

MODE_DICT_IDX_TO_MANEUVER = {0: "LT", 1: "RT", 2: "FF", 3: "SF", 4: "ST"}
MODE_DICT_MANEUVER_TO_IDX = {"LT": 0, "RT": 1, "FF": 2, "SF": 3, "ST": 4}

# The online definition adds additional maneuvers for forward (slow forward, normal forward, and fast forward).
MODE_DICT_IDX_TO_MANEUVER_ONLINE = {0: "LT", 1: "RT", 2: "ST", 3: "SF", 4: "NF", 5: "FF"}
MODE_DICT_MANEUVER_TO_IDX_ONLINE = {"LT": 0, "RT": 1, "ST": 2, "SF": 3, "NF": 4, "FF": 5}

MODE_DICT_IDX_TO_LANE_CHANGE = {0: "LK", 1: "LL", 2: "LR"}
MODE_DICT_LANE_CHANGE_TO_IDX = {"LK": 0, "LL": 1, "LR": 2}
