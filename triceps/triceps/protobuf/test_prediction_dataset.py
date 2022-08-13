import json
import os
from itertools import zip_longest
from operator import itemgetter
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from google.protobuf.json_format import MessageToJson, Parse, ParseDict
from google.protobuf.timestamp_pb2 import Timestamp

from intent.multiagents.cache_utils import compute_hash
from intent.multiagents.pedestrian_trajectory_prediction_util import prepare_pedestrian_model_params
from intent.multiagents.train_pedestrian_trajectory_prediction import parse_args
from intent.multiagents.training_utils import prepare_cache
from radutils.torch.torch_utils import apply_2d_coordinate_rotation_transform
from triceps.protobuf.prediction_dataset import ProtobufPredictionDataset
from triceps.protobuf.prediction_dataset_cache import CacheElement
from triceps.protobuf.prediction_training_pb2 import PredictionSet
from triceps.protobuf.proto_arguments import args_to_dict

TEST_NUM_TIMESTAMPS = 20


def valid_straight_trajectory(base_speed=torch.tensor((10.0, 0.0))) -> torch.Tensor:
    return torch.arange(TEST_NUM_TIMESTAMPS).unsqueeze(-1) * base_speed.unsqueeze(0)


def valid_timestamps(time_step=1.0) -> torch.Tensor:
    return torch.arange(TEST_NUM_TIMESTAMPS) * time_step


def timestamp_to_protobuf_string(timestamp: float):
    pb_timestamp = Timestamp()
    pb_timestamp.FromNanoseconds(int(timestamp * 1e9))
    return MessageToJson(pb_timestamp).strip('"')


def valid_agent_trajectory_dict(
    agent_id: int,
    is_ego: bool,
    timestamps: torch.Tensor = None,
    trajectory_local: torch.Tensor = None,
    rotation_scene_local: torch.Tensor = None,
    translation_scene_local: torch.Tensor = None,
):
    if trajectory_local is None:
        trajectory_local = valid_straight_trajectory()
    if rotation_scene_local is None:
        rotation_scene_local = torch.eye(2)
    if translation_scene_local is None:
        translation_scene_local = torch.zeros(2)
    if timestamps is None:
        timestamps = valid_timestamps()

    assert (
        trajectory_local.shape[0] == timestamps.shape[0]
    ), "trajectory_local and timestamps must have the same number of time points"

    trajectory_scene = apply_2d_coordinate_rotation_transform(
        rotations_a_b=rotation_scene_local,
        coordinates_b=trajectory_local,
        rotation_einsum_prefix="",
        result_einsum_prefix="t",
    )

    trajectory_scene += translation_scene_local + translation_scene_local.unsqueeze(0)

    additional_inputs = []
    if is_ego:
        # TODO replace with valid hash
        hash_prefix = "HASH_PREFIX"
        additional_inputs.append(dict(vectorInputTypeId="additional_input_egovehicle"))
        # Add image links
        additional_inputs.extend(
            dict(
                sensorImageInput=dict(
                    filename=json.dumps(
                        dict(
                            filename=f"{hash_prefix}_{i:07d}",
                            # TODO check how significant this field is
                            folder="fake/folder",
                        )
                    )
                ),
                timestamp=timestamp_to_protobuf_string(timestamp),
                vectorInputTypeId="additional_input_key_camera_image",
            )
            for i, timestamp in enumerate(timestamps)
        )
    else:
        additional_inputs.append(dict(vectorInputTypeId="additional_input_additional_agent"))

    return dict(
        agentId=str(agent_id),
        additionalInputs=additional_inputs,
        trajectory=[
            dict(
                timestamp=timestamp_to_protobuf_string(timestamp),
                position=dict(x=position[0], y=position[0]),
            )
            for timestamp, position in zip_longest(timestamps, trajectory_local)
        ],
    )


@pytest.fixture()
def valid_map_information():
    # Question: Are negative map coordinates ever valid?
    return dict(
        lanes=[
            dict(
                id="0",
                centerLine=[
                    dict(
                        start=dict(x=1.0, y=7.0),
                        end=dict(x=101.0, y=7.0),
                    )
                ],
                leftLaneBoundary=[
                    dict(
                        start=dict(x=1.0, y=13.0),
                        end=dict(x=101.0, y=13.0),
                    )
                ],
                rightLaneBoundary=[
                    dict(
                        start=dict(x=1.0, y=1.0),
                        end=dict(x=101.0, y=1.0),
                    )
                ],
            ),
        ],
        zones=[
            dict(
                id="crosswalk",
                type="Crosswalk",
                polygon=[
                    {"start": dict(x=15.0, y=6.0), "end": dict(x=20.0, y=6.0)},
                    {"start": dict(x=20.0, y=6.0), "end": dict(x=20.0, y=-6.0)},
                    {"start": dict(x=20.0, y=-6.0), "end": dict(x=15.0, y=-6.0)},
                ],
            )
        ],
    )


@pytest.fixture()
def valid_prediction_instance_dict(
    prediction_instance_info_json, prediction_time, valid_map_information, source_prediction_time
):
    return dict(
        agentTrajectories={
            # Moving past ego vehicle from opposite direction in opposite lane
            "31": valid_agent_trajectory_dict(
                agent_id=31,
                is_ego=False,
                translation_scene_local=torch.tensor((11.0, 10.0)),
                rotation_scene_local=torch.tensor(((-1.0, 0.0), (0.0, -1.0))),
            ),
            # Ego vehicle
            "-2": valid_agent_trajectory_dict(
                agent_id=-2, is_ego=True, translation_scene_local=torch.tensor((1.0, 4.0))
            ),
            # Fixed distance ahead of ego vehicle, same lane
            "13": valid_agent_trajectory_dict(
                agent_id=13, is_ego=False, translation_scene_local=torch.tensor((11.0, 4.0))
            ),
        },
        egovehicleId="-2",
        map_information=valid_map_information,
        predictionInstanceInfo=prediction_instance_info_json,
        predictionTime=prediction_time,
        sourcePredictionTime=source_prediction_time,
    )


@pytest.fixture()
def prediction_time() -> str:
    return "1970-01-01T00:00:00Z"


@pytest.fixture()
def source_prediction_time() -> str:
    return "2020-02-25T05:32:07.471716165Z"


@pytest.fixture()
def prediction_instance_info_json(source_prediction_time):
    prediction_timestamp = Parse(
        f'"{source_prediction_time}"',
        Timestamp(),
    )
    return json.dumps(
        dict(
            json_dir="some/json/dir",
            source_tlog="some/tlog/data.tlog",
            timestamps=str(prediction_timestamp.ToNanoseconds() / 1e9),
        )
    )


@pytest.fixture()
def valid_prediction_set(valid_prediction_instance_dict) -> PredictionSet:
    return ParseDict(
        dict(predictionInstances=[valid_prediction_instance_dict]),
        PredictionSet(),
    )


@pytest.fixture()
def valid_protobuf_file(tmp_path, valid_prediction_set):
    file = tmp_path / "valid_protobuf.pb"
    file.write_bytes(valid_prediction_set.SerializeToString())

    return str(file)


@pytest.fixture()
def model_params(tmp_path):
    # TODO(nicholas.guyett.ctr) Decouple this fixture from the intent module
    cache_dir = tmp_path / "cache"
    return prepare_cache(
        prepare_pedestrian_model_params(
            args_to_dict(
                parse_args(["--cache-dir", str(cache_dir)]),
                additional_file_path_params=["latent_factors_file"],
            )
        )
    )


class TestPredictionDataset:
    def test_loads_valid_protobuf(self, valid_protobuf_file, model_params):
        dataset = ProtobufPredictionDataset(
            filename=valid_protobuf_file,
            params=model_params,
        )

        # Invoke lazy validation.
        dataset.__getitem__(0)

        assert len(dataset) == dataset.num_total_instances

    def test_uses_cache_correctly(self, valid_protobuf_file, model_params):
        pb_index = 0
        cache_name = compute_hash(model_params["main_param_hash"] + f"parse_{valid_protobuf_file}_{pb_index}")
        cache_element = CacheElement(
            os.path.join(model_params["cache_dir"], "main_cache"),
            cache_name,
            "pkl",
            should_lock=model_params["use_cache_lock"],
            read_only=model_params["cache_read_only"],
            disable_cache=model_params["disable_cache"],
        )

        dataset = ProtobufPredictionDataset(
            filename=valid_protobuf_file,
            params=model_params,
        )
        assert not cache_element.is_cached()

        dataset_item = dataset[0]

        cached_dataset_item = cache_element.load()
        dataset._remove_unused_items(cached_dataset_item)  # Remove all entries not returned in the final result

        for key, cached_value in cached_dataset_item.items():
            returned_value = dataset_item[key]
            if np.isscalar(cached_value):
                assert returned_value == cached_value
            else:
                torch.testing.assert_equal(returned_value, cached_value)

    def test_parse_and_process_instance(
        self,
        valid_map_information,
        valid_prediction_instance_dict,
        valid_protobuf_file,
        model_params,
    ):
        """Verifies the protobuf parsing process"""
        mock_post_processing_key = "mock_post_processing"

        # Re-order trajectories to put the ego vehicle first
        sorted_trajectories = [
            trajectory
            for _, trajectory in sorted(valid_prediction_instance_dict["agentTrajectories"].items(), key=itemgetter(0))
        ]

        def check_processing_only_items(return_item, _params, _rel_filename, item):
            # Verify that the "processing only" fields are available to data transforms
            source_additional_inputs = [trajectory["additionalInputs"] for trajectory in sorted_trajectories]
            assert valid_map_information == return_item[ProtobufPredictionDataset.DATASET_KEY_MAP_INFORMATION]
            assert source_additional_inputs == return_item[ProtobufPredictionDataset.DATASET_KEY_ADDITIONAL_INPUTS]
            # TODO validate semantic data
            assert ProtobufPredictionDataset.DATASET_KEY_SEMANTIC_TARGETS in return_item

            return_item[mock_post_processing_key] = mock_post_processing_key

        mock_transform = MagicMock()
        mock_transform.process = check_processing_only_items

        self.dataset = ProtobufPredictionDataset(
            valid_protobuf_file,
            data_transforms=[mock_transform],
            params=model_params,
        )

        final_result = self.dataset[0]

        # "processing only" keys should not be available downstream from the dataset
        assert ProtobufPredictionDataset.DATASET_KEY_MAP_INFORMATION not in final_result
        assert ProtobufPredictionDataset.DATASET_KEY_ADDITIONAL_INPUTS not in final_result
        assert ProtobufPredictionDataset.DATASET_KEY_SEMANTIC_TARGETS not in final_result

        # Fields added by data handlers should be present in the final result
        assert final_result[mock_post_processing_key] == mock_post_processing_key

        # Validate trajectories
        def get_position_validity(pos):
            return 1 if pos["position"]["x"] or pos["position"]["y"] else 0

        source_trajectories = torch.Tensor(
            [
                [
                    [pos["position"]["x"], pos["position"]["y"], get_position_validity(pos)]
                    for pos in source_trajectory["trajectory"]
                ]
                for source_trajectory in sorted_trajectories
            ]
        )

        torch.testing.assert_equal(
            torch.Tensor(final_result[ProtobufPredictionDataset.DATASET_KEY_POSITIONS]),
            source_trajectories,
        )

    # TODO add test cases for "invalid" protobuf files
    # TODO expand to stubs and hash methods?
