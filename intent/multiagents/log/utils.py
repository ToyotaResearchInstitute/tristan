import logging

import wandb
import wandb.errors

from intent.multiagents.log.trainer_logging import NullLogger, TrainingLogger
from intent.multiagents.log.trainer_logging_tensorboard import DictLogger, TensorboardTrainingLogger
from intent.multiagents.log.trainer_logging_wandb import WandbTrainingLogger


def get_training_logger(session_name, device, params) -> TrainingLogger:
    if params["logger_type"] == "tensorboard":
        training_logger = TensorboardTrainingLogger(params, device.type, session_name=session_name)
    elif params["logger_type"] == "weights_and_biases":
        try:
            training_logger = WandbTrainingLogger(
                params, device.type, session_name=session_name, mode=params["wandb_mode"]
            )
        except (Exception, wandb.errors.Error) as e:
            logging.error(f"Wandb init failed, logging offline, error: {e}")
            print(f"Wandb init failed, logging offline, error: {e}")
            wandb.finish()
            if hasattr(wandb, "teardown"):
                wandb.teardown()
            training_logger = WandbTrainingLogger(params, device.type, session_name=session_name, mode="offline")
        pass

    elif params["logger_type"] == "dict":
        training_logger = DictLogger()
    elif params["logger_type"] == "none":
        training_logger = NullLogger(params, session_name=session_name)
    else:
        raise RuntimeError("Invalid --logger-type")
    return training_logger
