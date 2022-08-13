import random

import numpy as np
import torch

from radutils.misc import get_current_branch, get_current_commit_hash
from radutils.reproducibility import WANDB_URL, RandomState, RunManifest, seed_everything


def test_random_state():
    seed_everything(0)

    current_state = RandomState()

    num_count = 10

    expected_random_python_numbers = [random.random() for _ in range(num_count)]
    expected_random_numpy_numbers = np.random.random(num_count)
    expected_random_torch_numbers = torch.rand(num_count)

    deserialized_state = RandomState.deserialize(current_state.serialize())
    deserialized_state.apply()

    actual_random_python_numbers = [random.random() for _ in range(num_count)]
    actual_random_numpy_numbers = np.random.random(num_count)
    actual_random_torch_numbers = torch.rand(num_count)

    assert actual_random_python_numbers == expected_random_python_numbers
    np.testing.assert_equal(actual_random_numpy_numbers, expected_random_numpy_numbers)
    assert torch.equal(actual_random_torch_numbers, expected_random_torch_numbers)


def test_run_manifest_serialization():
    session_name = "test_run_manifest_serialization"
    fake_params = {
        "logger_type": None,
        "subdict": {
            "foo": "bar",
        },
        "number:": 2,
        "fish_list": [1, 2, "red", "blue"],
        "provided_args": ["do_train", "--some", "other", "ar=gs"],
    }
    run_manifest = RunManifest(session_name, fake_params)

    expected_serialization = {
        "session_name": session_name,
        "git_info": {
            "branch": get_current_branch(),
            "hash": get_current_commit_hash(),
        },
        "provided_args": fake_params["provided_args"],
    }
    assert run_manifest.serialize() == expected_serialization


def test_run_manifest_serialization_with_wandb():
    session_name = "test_run_manifest_serialization_with_wandb"
    wandb_entity = "test"
    wandb_project = "wandb"
    fake_params = {
        "logger_type": "weights_and_biases",
        "wandb_entity": wandb_entity,
        "wandb_project": wandb_project,
        "subdict": {
            "foo": "bar",
        },
        "number:": 2,
        "fish_list": [1, 2, "red", "blue"],
        "provided_args": ["do_train", "--some", "other", "ar=gs"],
    }
    run_manifest = RunManifest(session_name, fake_params)

    expected_serialization = {
        "session_name": session_name,
        "git_info": {
            "branch": get_current_branch(),
            "hash": get_current_commit_hash(),
        },
        "provided_args": fake_params["provided_args"],
        "wandb_results_link": f"{WANDB_URL}/{wandb_entity}/{wandb_project}/runs/{session_name}",
    }
    assert run_manifest.serialize() == expected_serialization
