"""
Default data key names for json files of ado cars and other track objects.
"""

## information for current data sample
FILEHASH = "filehash"
FILEDIR = "file_dir"
SOURCE_CSV = "source_csv"
# id of target (reference) vehicle
UNIQUE_ID = "unique_id"
# count of sample extracted from the same dataset
SAMPLE_COUNT = "sample_count"
# count of data sequence for the same vehicle
SEQUENCE_COUNT = "sequence_count"
# order within a sequence
SEQUENCE_ORDER = "sequence_order"
TIMESTAMP = "timestamp"
TIMESTAMPED_DATA = "timestamped_data"

## target (reference) vehicle info
POSITION_L = "position_L"
HEADING_L = "heading_L"
MANEUVER_MODE = "maneuver_mode"
LANE_INDEX = "lane_index"
# The smoothed index provides cleaner labels when an agent follows a lane but crosses other lanes.
LANE_INDEX_SMOOTHED = "lane_index_smoothed"
LANE_CHANGE_MODE = "lane_change_mode"
# Language description for the trajectory
LANGUAGE_TOKENS = "language_tokens"
TOKEN = "token"
MAIN_AGENT = "main_agent_id"
OTHER_AGENT = "other_agent_id"
START_TIMESTAMP = "start_timestamp"
END_TIMESTAMP = "end_timestamp"
## nearby agents info
NEARBY_AGENTS = "nearby_agents"

# map info
LANES = "lanes"
CITY = "city"
IMAGE_INPUTS = "image_inputs"

# type of target vehicle
DFS_VEH_TYPE = "dfs_veh_type"

# caching info
JSON_LANES_CACHE = "json_lanes_cache"

PEDESTRIAN_INTENT_LABEL = "pedestrian_intent"
PEDESTRIAN_INTENT_TIMESTAMP_LABEL = PEDESTRIAN_INTENT_LABEL + "_timestamp"
PREDICTION_TIMESTAMP = "prediction_timestamp"

ZONES = "zones"

# mimic track_semantics
AGENT_TYPE_UNKNOWN = 0
AGENT_TYPE_STATIC = 1
AGENT_TYPE_CAR = 2
AGENT_TYPE_BICYCLE = 3
AGENT_TYPE_MOTORCYCLE = 4
AGENT_TYPE_PEDESTRIAN = 5
AGENT_TYPE_LARGEVEHICLE = 6
AGENT_TYPE_TRUCK = 7
AGENT_TYPE_NAME_MAP = {
    AGENT_TYPE_UNKNOWN: "unknown",
    AGENT_TYPE_STATIC: "static",
    AGENT_TYPE_CAR: "car",
    AGENT_TYPE_BICYCLE: "bicycle",
    AGENT_TYPE_MOTORCYCLE: "motorcycle",
    AGENT_TYPE_PEDESTRIAN: "pedestrian",
    AGENT_TYPE_LARGEVEHICLE: "largevehicle",
    AGENT_TYPE_TRUCK: "truck",
}
PAST_COEFFICIENTS = "past_coefficients"
PAST_TRAJECTORY = "past_trajectories"
PAST_EGOVEHICLE_TRAJECTORY = "past_egovehicle_trajectory"
FUTURE_COEFFICIENTS = "future_coefficients"
FUTURE_TRAJECTORY = "future_trajectories"
FUTURE_EGOVEHICLE_TRAJECTORY = "future_egovehicle_trajectory"
PEDESTRIAN_INTENT_VECTOR = "pedestrian_intent_vector"
PEDESTRIAN_AWARENESS_VECTOR = "pedestrian_awareness_vector"
CROSSING_PROBABILITY = "crossing_probability"
WEIGHTED_INTENT = "weighted_intent"
CROSSWALK_DISTANCE = "crosswalk_distance"
RIGHT_OF_WAY = "has_right_of_way"
OBJECT_IMAGE = "object_image"
MAP_FEATURES = "map_features"
MAP_FEATURES_DICT = "map_features_dictionary"
NUM_PA_DETECTIONS = "NUM_PA_DETECTIONS"
MIN_LANE_DISTANCE = "MIN_LANE_DISTANCE"
NEAREST_LANE_POSITION0 = "NEAREST_LANE_POSITION0"
NEAREST_LANE_POSITION1 = "NEAREST_LANE_POSITION1"
MIN_EGO_LANE_DISTANCE = "MIN_EGO_LANE_DISTANCE"
MIN_EGO_ZONE_DISTANCE = "MIN_EGO_ZONE_DISTANCE"
EGO_ADO_DISTANCE = "EGO_ADO_DISTANCE"
MIN_ZONE_DISTANCE = "MIN_ZONE_DISTANCE"
NEAREST_ZONE_POSITION0 = "NEAREST_ZONE_POSITION0"
NEAREST_ZONE_POSITION1 = "NEAREST_ZONE_POSITION1"
RIGHT_OF_WAY_CLOSEST = "RIGHT_OF_WAY_CLOSEST"
IS_LEAKED = "is_leaked"
