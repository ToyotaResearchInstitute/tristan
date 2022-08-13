import logging
from abc import ABC, abstractmethod
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np


class TrainingLogger(ABC):
    """Logger interface class for both training statistics and auxiliary text message.

    Parameters
    ----------
    params: dict
      A parameter dictionary. Contains 'logs_dir' for the folder name.

    session_name: str
      A session name. Will be used to set a log folder.

    """

    @abstractmethod
    def __init__(self, params: dict, session_name: str) -> None:
        """Initializes the logger."""

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    def epoch_end(self, epoch: int, global_batch_cnt: int) -> None:
        """Logic to execute at the end of an epoch.

        Parameters
        ----------
        epoch: int
          The currently ending epoch
        global_step: int
          The global step to use.
        """


class MessageLogger(ABC):
    """
    Class to save log messages -- any error level, similar to python's logging module.
    """

    @abstractmethod
    def log_message(self, message: str, error_level: Optional[int] = logging.INFO):
        """

        Parameters
        ----------
        message: str
          Message text to save.
        error_level: Optional[int]
          The error level (uses python's logging levels).

        """


class TerminalMessageLogger(MessageLogger):
    """
    Class to save log messages -- currently prints to screen.
    """

    def log_message(self, message: str, error_level: Optional[int] = logging.INFO) -> None:
        """

        Parameters
        ----------
        message: str
          Message text to save.
        error_level: Optional[int]
          The error level (uses python's logging levels). (currently unused)

        """
        print(message)
