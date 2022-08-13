import numbers
import os
import random
from collections.abc import Iterable
from typing import Any, Dict, NamedTuple, TypedDict, Union, cast

import numpy as np
import torch

from radutils.misc import get_current_branch, get_current_commit_hash
from radutils.torch.async_saver import SAVED_ARTIFACT_S3_PREFIX, SAVED_MODEL_S3_PREFIX

WANDB_URL = "https://wandb.ai"


def seed_everything(seed: int):
    """Manually set all relevant random seeds.

    Based on https://github.com/PyTorchLightning/PyTorch-Lightning/blob/0.8.3/pytorch_lightning/utilities/seed.py

    Args:
    :param seed: The random seed to be used in all random number generators.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SerializedRandomState(TypedDict):
    env_seed: str
    python_state: list
    numpy_state: list
    torch_state: list


class NumpyRandomState(NamedTuple):
    algorithm: str
    keys: np.ndarray
    pos: int
    has_gauss: int
    cached_gaussian: float

    @staticmethod
    def deserialize(algorithm: str, keys: Iterable, pos: int, has_gauss: int, cached_gaussian: float):
        return NumpyRandomState(
            algorithm=algorithm,
            keys=np.array(keys),
            pos=pos,
            has_gauss=has_gauss,
            cached_gaussian=cached_gaussian,
        )


def deserialize_python_state(state: Iterable):
    return tuple(tuple(val) if isinstance(val, Iterable) else val for val in state)


class RandomState:
    @staticmethod
    def _serialize_state(val) -> Union[list, None, numbers.Number, str]:
        if hasattr(val, "tolist"):
            # ndarray or tensor
            return val.tolist()
        elif isinstance(val, tuple):
            return list(map(RandomState._serialize_state, val))
        else:
            return val

    def __init__(self, state_dict: Dict = None):
        if state_dict:
            self.env_seed = str(state_dict["env_seed"])
            self.python_state: tuple = deserialize_python_state(state_dict["python_state"])
            self.numpy_state = NumpyRandomState.deserialize(*state_dict["numpy_state"])
            self.torch_state = torch.ByteTensor(state_dict["torch_state"])
        else:
            self.env_seed: str = os.environ.get("PYTHONHASHSEED", None)
            self.python_state: tuple = cast(tuple, random.getstate())
            self.numpy_state: NumpyRandomState = NumpyRandomState(*np.random.get_state())
            self.torch_state = torch.random.get_rng_state()

    def apply(self):
        """Sets random state in current environment"""
        os.environ["PYTHONHASHSEED"] = self.env_seed
        random.setstate(self.python_state)
        np.random.set_state(cast(Any, self.numpy_state))  # numpy accepts a tuple, but type checking doesn't know that
        torch.random.set_rng_state(self.torch_state)

    def serialize(self) -> Dict:
        return SerializedRandomState(
            env_seed=self.env_seed,
            python_state=list(self._serialize_state(self.python_state)),
            numpy_state=list(self._serialize_state(self.numpy_state)),
            torch_state=self.torch_state.tolist(),
        )

    @staticmethod
    def deserialize(state_dict: Dict):
        return RandomState(state_dict)


class RunManifest:
    def __init__(self, session_name: str, params: dict, preserve_rng: bool = False):
        super().__init__()
        self.session_name = session_name
        self.provided_args = params["provided_args"]
        self.git_info = {
            "branch": get_current_branch(),
            "hash": get_current_commit_hash(),
        }
        self.random_state = RandomState() if preserve_rng else None

        if params["logger_type"] == "weights_and_biases":
            self.wandb_results_link = (
                f"{WANDB_URL}/{params['wandb_entity']}/{params['wandb_project']}/runs/{self.session_name}"
            )
        else:
            self.wandb_results_link = None

    def serialize(self) -> dict:
        result = {
            "session_name": self.session_name,
            "git_info": self.git_info,
            "provided_args": self.provided_args,
        }

        if self.random_state:
            result["random_state"] = self.random_state

        if self.wandb_results_link:
            result["wandb_results_link"] = self.wandb_results_link

        return result
