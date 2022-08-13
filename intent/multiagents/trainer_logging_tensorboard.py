import os
from typing import Optional, Union

import numpy as np
import nvidia_smi
import psutil
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from intent.multiagents.trainer_logging import TrainingLogger


class DictLogger(TrainingLogger):
    """Record all data logged to it, so it can be pickled and logged to Tensorboard/wandb logger.
    This is necessary because loggers are not picklable, so can't use across process.
    """

    def __init__(self):
        self.images = []
        self.texts = []
        self.figs = []
        self.scalars = []
        self.hists = []

    @staticmethod
    def _add(variable: list, args: tuple, kwargs: dict):
        """Record the variable and it's calling arguments.
        Parameters:
        variable: list
            The variable which is used to record the calling arguments
        args: tuple
            The positional arguments
        kwargs: dict
            The key word arguments
        """
        variable.append((args, kwargs))

    def add_text(self, *args, **kwargs) -> None:
        self._add(self.texts, args, kwargs)

    def add_image(self, *args, **kwargs) -> None:
        self._add(self.images, args, kwargs)

    def add_figure(self, *args, **kwargs) -> None:
        self._add(self.figs, args, kwargs)

    def add_scalar(self, *args, **kwargs) -> None:
        self._add(self.scalars, args, kwargs)

    def add_histogram(self, *args, **kwargs) -> None:
        self._add(self.hists, args, kwargs)

    def log_to_real_logger(self, logger: TrainingLogger):
        """Log the recorded info to a real logger"""
        for t in self.texts:
            # unpack the calling arguments.
            logger.add_text(*t[0], **t[1])
        for t in self.images:
            logger.add_image(*t[0], **t[1])
        for t in self.figs:
            logger.add_figure(*t[0], **t[1])
        for t in self.scalars:
            logger.add_scalar(*t[0], **t[1])
        for t in self.hists:
            logger.add_histogram(*t[0], **t[1])


class TensorboardTrainingLogger(TrainingLogger):
    """Logger interface class for both training statistics and auxiliary text message.
    Uses Tensorboard.

    Parameters
    ----------
    params: dict
      A parameter dictionary. Contains 'logs_dir' for the folder name.

    session_name: str
      A session name. Will be used to set a log folder.
    """

    def __init__(self, params: dict, device_type: str, session_name: str) -> None:
        """Initializes a Tensorboard-based logger"""
        self.session_name = session_name
        self.log_folder = os.path.expanduser(os.path.join(params["logs_dir"], self.session_name))
        self.writer = SummaryWriter(log_dir=self.log_folder, comment="GNN training")
        self.device_type = device_type

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
        self.writer.add_text(tag, text_string, global_step)

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
        self.writer.add_image(tag, img, global_step)

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
        self.writer.add_figure(tag, figure, global_step)

    def add_scalar(
        self, tag: str, scalar_value: Union[float, np.number, np.ndarray], global_step: Optional[int] = None
    ) -> None:
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
        self.writer.add_scalar(tag, scalar_value, global_step)

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
        self.writer.add_histogram(tag, values, global_step, bins)

    def epoch_end(self, epoch: int, global_batch_cnt: int) -> None:
        """Logic to execute at the end of an epoch.

        Parameters
        ----------
        epoch: int
          The currently ending epoch
        global_step: int
          The global step to use.
        """
        process = psutil.Process(os.getpid())
        self.add_scalar("sys/memory_usage", process.memory_info().rss, global_step=global_batch_cnt)
        self.add_scalar("sys/number_of_threads", process.num_threads(), global_step=global_batch_cnt)

        if "cuda" in self.device_type:
            nvidia_smi.nvmlInit()
            device_id = torch.cuda.current_device()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_id)
            gpu_memory = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            gpu_rates = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            self.add_scalar(
                "sys/gpu_memory_usage_ratio", gpu_memory.used / gpu_memory.total, global_step=global_batch_cnt
            )
            self.add_scalar("sys/gpu_memory_free", gpu_memory.free, global_step=global_batch_cnt)
            self.add_scalar("sys/gpu_proc_usage", gpu_rates.gpu / 100.0, global_step=global_batch_cnt)
