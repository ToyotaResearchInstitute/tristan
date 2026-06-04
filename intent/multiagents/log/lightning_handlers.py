from typing import List, Union, cast

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT

from intent.multiagents.log.logging_handlers import LoggingHandler
from intent.multiagents.log.trainer_logging import NullLogger, TrainingLogger
from intent.multiagents.log.trainer_logging_tensorboard import DictLogger
from intent.multiagents.log.trainer_logging_wandb import WandbTrainingLogger
from triceps.protobuf.prediction_dataset import ProtobufPredictionDataset


def create_lightning_logger(params):
    if params["logger_type"] == "none":
        return NullLogger(params, "")

    if params["logger_type"] == "tensorboard":
        logger = TensorBoardLogger(params["cache_dir"], params["current_session_name"])
    elif params["logger_type"] == "weights_and_biases":
        logger = WandbLogger(
            params["current_session_name"],
            params["cache_dir"],
        )
        logger.experiment.config.update(params)
    else:
        raise ValueError(f"Logger type {params['logger_type']} is not available")

    return logger


class FakeWandbLogger(WandbTrainingLogger):
    """Exposes the intent WandbTrainingLogger logic and interface without actually initializing a wandb run"""

    # noinspection PyMissingConstructor
    def __init__(self):
        self._cur_log = {}

    def get_log_dictionary(self):
        return self._cur_log


class ProxyLogger(DictLogger):
    """Wraps pytorch-lightning's logging api as if it were a legacy intent logger"""

    def log_to_real_logger(self, logger: LightningLoggerBase):
        if isinstance(logger, TensorBoardLogger):
            # TrainingLogger is based on tensorboard's summary writer, so we can pass it directly
            super().log_to_real_logger(cast(TrainingLogger, logger.experiment))
        elif isinstance(logger, WandbLogger):
            fake_logger = FakeWandbLogger()
            super().log_to_real_logger(fake_logger)

            logger.experiment.log(fake_logger.get_log_dictionary())
        # else: Ignore all accrued logs
        # No need to clean-up, because this is destroyed and recreated every epoch


class ProxyHandler(Callback):
    """Wraps legacy intent logging handlers so they can be used as callbacks for pytorch-lightning"""

    def __init__(self, logging_handler: LoggingHandler, params):
        super().__init__()
        self.logging_handler = logging_handler
        self.params = params

    def on_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Start epoch with a fresh proxy logger
        self.logging_handler.initialize_training(ProxyLogger())

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.logging_handler.epoch_start("train")

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.logging_handler.epoch_start("validation")

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.logging_handler.epoch_end(trainer.current_epoch, trainer.global_step)
        cast(ProxyLogger, self.logging_handler.writer).log_to_real_logger(trainer.logger)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        num_past_time_points = batch[ProtobufPredictionDataset.DATASET_KEY_NUM_PAST_POINTS]

        full_trajectory_scene = batch[ProtobufPredictionDataset.DATASET_KEY_POSITIONS]
        actual_future_trajectory_scene = full_trajectory_scene[:, :, num_past_time_points:]
        valid_positions = full_trajectory_scene[..., -1]

        additional_inputs = batch[ProtobufPredictionDataset.DATASET_KEY_ADDITIONAL_INPUTS]

        relevant_agents = batch[ProtobufPredictionDataset.DATASET_KEY_IS_RELEVANT_AGENT]
        agent_additional_inputs = []  # TODO Look up how to generate this
        agent_types = batch[ProtobufPredictionDataset.DATASET_KEY_AGENT_TYPE]

        data_dictionary = {
            "input_tensor": full_trajectory_scene,
            "additional_inputs": additional_inputs,
            "agent_additional_inputs": agent_additional_inputs,
            "predicted_trajectories": outputs["prediction"],
            "expected_trajectories": actual_future_trajectory_scene,
            "expected_trajectories_full": full_trajectory_scene,
            "is_valid": valid_positions[:num_past_time_points],
            "is_future_valid": valid_positions[num_past_time_points:],
            "is_valid_full": valid_positions,
            "batch_cost": outputs["loss"],
            "batch_itm": batch,
            "batch_itm_index": batch_idx,
            "batch_agent_type": agent_types,
            "relevant_agents": relevant_agents,
            # "offset_x": offset_x,
            # "offset_y": offset_y,
            "param": self.params,
            "stats_list": outputs["stats"],
            "global_batch_cnt": trainer.global_step,
            "agent_transforms": outputs["stats"][0]["agent_transforms"],
        }
        self.logging_handler.iteration_update(
            data_dictionary,
            {"g_stats": outputs["stats"]},
        )


def logging_handlers_as_lightning_callbacks(params, handlers: List[Union[LoggingHandler, Callback]]):
    return [handler if isinstance(handler, Callback) else ProxyHandler(params, handler) for handler in handlers]
