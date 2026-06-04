from typing import Tuple, Type

from torch.utils.data import Dataset

from intent.multiagents.additional_costs import TrajectoryRegularizationCost
from intent.multiagents.latent_factors import LatentFactorsCallback
from intent.multiagents.logging_handlers import LogWorstCases, WaymoLogHandler
from intent.multiagents.pedestrian_trajectory_prediction_util import BENCHMARK_GPU_NUM_EPOCHS
from model_zoo.intent.prediction_model import PredictionModel
from model_zoo.intent.prediction_model_interface import PredictionModelInterface
from triceps.protobuf.prediction_dataset import PredictionDataset, ProtobufPredictionDataset

try:
    from intent.multiagents.logging_handlers import GPUUtilizationLogHandler
except ImportError:
    GPUUtilizationLogHandler = None


class TrainingConfig:
    """Configuration shim for scripts that need dataset/cache metadata.

    The private training stack has a larger TrainingConfig object. The public
    release only needs these fields for data loading and filtering scripts.
    """

    dataset_cls: Type[Dataset] = PredictionDataset
    prediction_instance_cls: Type[ProtobufPredictionDataset] = ProtobufPredictionDataset
    model_cls: Type[PredictionModelInterface] = PredictionModel
    trainer_cls = None
    training_strategy_cls = None
    log_manager_cls = None
    datasets_filter_cls = None

    additional_log_tensor_agent_keys: Tuple[str, ...] = ()
    additional_log_tensor_batch_keys: Tuple[str, ...] = ()
    additional_log_list_keys: Tuple[str, ...] = ()
    additional_metric_list: Tuple[str, ...] = ()
    additional_cache_param_keys: Tuple[str, ...] = ()

    additional_logging_handlers: Tuple = ()
    additional_model_callbacks: Tuple = ()
    additional_trainer_callbacks: Tuple = ()
    custom_vis_callbacks: Tuple = ()
    latent_factors_trainer_callback = None

    def __init__(
        self,
        additional_logging_handlers=None,
        additional_model_callbacks=None,
        additional_trainer_callbacks=None,
        custom_vis_callbacks=None,
        latent_factors_trainer_callback=None,
    ):
        self.additional_logging_handlers = tuple(additional_logging_handlers or ())
        self.additional_model_callbacks = tuple(additional_model_callbacks or ())
        self.additional_trainer_callbacks = tuple(additional_trainer_callbacks or ())
        self.custom_vis_callbacks = tuple(custom_vis_callbacks or ())
        self.latent_factors_trainer_callback = latent_factors_trainer_callback

    @classmethod
    def from_params(cls, params: dict) -> "TrainingConfig":
        additional_logging_handlers = []
        additional_model_callbacks = []
        latent_factors_trainer_callback = None

        if params.get("use_semantics"):
            latent_factors_trainer_callback = LatentFactorsCallback(params)

        if params.get("trajectory_regularization_cost", 0) > 0:
            additional_model_callbacks.append(TrajectoryRegularizationCost())

        if params.get("benchmark_gpu"):
            if GPUUtilizationLogHandler is None:
                raise RuntimeError("GPU benchmark logging is not available in this public release")
            print(f"Running GPU benchmark; setting --num-epochs to {BENCHMARK_GPU_NUM_EPOCHS}")
            params["num_epochs"] = BENCHMARK_GPU_NUM_EPOCHS
            params["logger_type"] = "tensorboard"
            additional_logging_handlers.append(GPUUtilizationLogHandler(params))

        if not params.get("use_simple_metrics", False):
            worst_cases_folder = params["runner_output_folder"] if params.get("logger_type") == "none" else None
            additional_logging_handlers.extend(
                [
                    LogWorstCases("g_cost", params, worst_cases_folder),
                    LogWorstCases("distance_x0", params, worst_cases_folder),
                    LogWorstCases("fde_agent", params, worst_cases_folder),
                ]
            )

        if params.get("use_waymo_dataset") and params.get("report_waymo_metrics"):
            additional_logging_handlers.append(WaymoLogHandler(params))

        if params.get("use_waymo_dataset"):
            params["agent_types"] = [0, 2, 5, 3, 8]

        return cls(
            additional_logging_handlers=additional_logging_handlers,
            additional_model_callbacks=additional_model_callbacks,
            latent_factors_trainer_callback=latent_factors_trainer_callback,
        )


class LegacyTrainingConfig(TrainingConfig):
    pass
