import copy
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from intent.multiagents.trainer_logging import TrainingLogger


def load_wandb_config(params, wandb_config):
    if not params["use_wandb_config"]:
        return
    # Overrid param if defined in wandb
    for k in dict(wandb_config).keys():
        if k in params and params[k] != wandb_config[k]:
            print(f"Override param[{k}] from {params[k]} with {wandb_config[k]}")
            params[k] = wandb_config[k]


def should_resume_wandb_session(params: Dict) -> bool:
    current = params.get("current_session_name")
    resume = params.get("resume_session_name")
    return current and resume and current == resume


class WandbTrainingLogger(TrainingLogger):
    """Weights and biases logger interface.

    Parameters
    ----------
    params: dict
      A parameter dictionary. Contains 'logs_dir' for the folder name.

    session_name: str
      A session name. Will be used to set a log folder.
    """

    # NOTE: Most of the following keys should not be in the parameter dictionary to begin with.
    # Thus, we exclude them from logging. This can be removed once param is refactored / cleaned-up. Furthermore,
    # "datasets_name_lists" is excluded because it refers to a json file which is automatically loaded due to some
    # wandb magic. As this file is quite big (containing our full split), we want to avoid this.
    _EXCLUDE_PARAM_KEYS = [
        "latent_factors_generator",
        "datasets_name_lists",
        "latent_factors",
    ]

    def __init__(self, params: dict, device_type: str, session_name: str) -> None:
        """Initializes a Tensorboard-based logger w/ Weights and Biases."""
        super().__init__(params, session_name)
        self._cur_log = {}
        self.device_type = device_type

        params_orig = copy.deepcopy(params)
        run = wandb.init(
            config=params,
            project=params["wandb_project"],
            entity=params["wandb_entity"],
            id=session_name,
            config_exclude_keys=self._EXCLUDE_PARAM_KEYS,
            resume=should_resume_wandb_session(params),
        )
        # Override CLI params if needed.
        load_wandb_config(params_orig, run.config)

    def add_text(self, tag: str, text_string: str, global_step: Optional[int] = None) -> None:
        """Log text.

        Parameters
        ----------
        tag: str
          Tag/label to store the data by.
        text_string: str
          Text to store.
        global_step: Optional[int]
          The iteration to store data for.
        """
        self._cur_log[tag] = [wandb.Html(data=text_string, inject=False)]

    def add_image(self, tag: str, img: np.array, global_step: Optional[int] = None) -> None:
        """Log an image.

        Parameters
        ----------
        tag: str
          Tag/label to store the data by.
        img: np.array
          Image to store.
        global_step: Optional[int]
          The iteration to store data for.

        """
        # We use images in shape [num_colors x height x width]. Weights and biases
        # likes the color dimension to be at the end.
        try:
            if isinstance(img, torch.Tensor):
                img_transposed = img.transpose(1, 0).transpose(1, 2).cpu().detach().numpy()
            else:
                img_transposed = img.transpose(1, 2, 0)
            self._cur_log[tag] = [wandb.Image(img_transposed)]
        except:
            import IPython

            IPython.embed(header="wandb logger")

    def add_figure(self, tag: str, figure: plt.Figure, global_step: Optional[int] = None) -> None:
        """Log a figure.

        Parameters
        ----------
        tag: str
          Tag/label to store the data by.
        figure: plt.Figure
          Figure to store.
        global_step: Optional[int]
          The iteration to store data for.

        """
        self._cur_log[tag] = [wandb.Image(figure)]

    def add_scalar(
        self, tag: str, scalar_value: Union[float, np.number, np.ndarray], global_step: Optional[int] = None
    ):
        """Log numerical statistics.

        Parameters
        ----------
        tag: str
          Tag/label to store the data by.
        scalar_value: Union[float, np.number, np.ndarray]
          Value to store.
        global_step: Optional[int]
          The iteration to store data for.
        """
        self._cur_log[tag] = scalar_value

    def add_histogram(
        self, tag: str, values: np.ndarray, global_step: Optional[int] = None, bins: Optional[str] = "tensorflow"
    ) -> None:
        """Log histogram statistics.

        Parameters
        ----------
        tag: str
          Tag/label to store the data by.
        values: np.ndarray
          histogram vector to store.
        global_step: Optional[int]
          The iteration to store data for.
        bins: Optional[str]
          bin vector.
        """
        self._cur_log[tag] = wandb.Histogram(values)

    def epoch_end(self, epoch: int, global_batch_cnt: int) -> None:
        """Logic to execute at the end of an epoch.

        Parameters
        ----------
        epoch: int
          The currently ending epoch
        global_step: int
          The global step to use.
        """
        wandb.log(self._cur_log)
        self._cur_log = {}
