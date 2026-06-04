from abc import ABC, abstractmethod
from typing import Callable, Iterable, List, Optional, Tuple

import torch
import wandb

from triceps.protobuf.prediction_dataset import collate_fn_with_padding


def log_image(logger, save_name, image, step):
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

    if isinstance(logger, TensorBoardLogger):
        img = image.transpose(2, 0, 1)
        logger.experiment.add_image(save_name, img, step)
    elif isinstance(logger, WandbLogger):
        logger.experiment.log({save_name: wandb.Image(image)})


class LightingLoggingHandler(ABC):
    """Generic Logging Interface for PyTorch Lighting"""

    def __init__(self, params: Optional[dict], config) -> None:
        """
        step_type: str
            The step type of this logger, 'training' or 'validation'
            Any update not the same as the step_type will be ignored.
        """
        super().__init__()
        if params is None:
            params = {}
        self.params = params
        self.config = config
        self.writer = None
        self.step_type = None

    def initialize_training(self, writer):
        self.writer = writer

    def epoch_start(self, dataloader_type: str, data) -> None:
        """Logger logic triggered at the beginning of an epoch.

        Parameters
        ----------
        dataloader_type : str
            The dataloader type, training/validation, etc.
        """
        self.data = data
        self.step_type = dataloader_type

    @abstractmethod
    def iteration_update(self, step_type, batch: dict, other: dict) -> None:
        """Logger logic triggered at each training iteration.

        Parameters
        ----------
        step_type : str
            The step type of this update, 'training' or 'validation'
        batch : dict
            All batch data and relevant intermediate data
        other : dict
            currently unused.
        """
        assert step_type == self.step_type, (
            f"The step: {step_type} isn't the same as self.step_type: {self.step_type}. "
            f"Porbably didn't call self.epoch_start/epoch_end!!"
        )

    def epoch_end(self, idx: int, global_batch_cnt: int, skip_visualization: bool) -> None:
        """Logger logic triggered at the end of an epoch.

        Parameters
        ----------
        idx : int
            Current epoch id.

        global_batch_cnt : int
            Number of batches processed so far.

        skip_visualization: bool
            Flag to toggle running of visualization code.
        """
        self.step_type = None


class GenericScalarLogger(LightingLoggingHandler):
    def __init__(self, metric_keys: Iterable[str], operator=None, **kwargs):
        super(GenericScalarLogger, self).__init__(**kwargs)
        self.logger_name = metric_keys
        self.metric_keys = metric_keys
        # Store the values of the metric
        self.values = {}
        self.torch_operator = torch.mean if operator is None else operator

    def collect_value(self, data_dictionary: dict):
        values = {}
        for key in self.metric_keys:
            if key not in data_dictionary:
                continue
            val = data_dictionary[key]

            if isinstance(val, torch.Tensor):
                assert len(val.shape) <= 1
                val = val.detach().cpu()
            else:
                val = torch.as_tensor(val)
            values[key] = val
        return values

        records = {k: data_dictionary[k] for k in self.recording_keys}
        sorted_values, indices, records = self.keep_max_values(val, records)
        return sorted_values, indices, records

    def epoch_start(self, dataloader_type: str, data):
        super().epoch_start(dataloader_type, data)
        self.values = {}

    def iteration_update(self, step_type, batch: dict, other: dict) -> None:
        super().iteration_update(step_type, batch, other)
        values = self.collect_value(batch)

        if values:
            for key, val in values.items():
                if key not in self.values:
                    self.values[key] = [val]
                else:
                    self.values[key].append(values[key])

    def epoch_end(self, idx: int, global_batch_cnt: int, skip_visualization: bool) -> None:
        if not self.values:
            return
        for key, value in self.values.items():
            if value[0].dim() == 0:
                val = torch.as_tensor(value)
            else:
                val = torch.cat(value)

            # Take the mean/max
            val_aggregate = self.torch_operator(val).item()

            self.writer.add_scalar(f"{self.step_type}_{key}", val_aggregate, idx)
            self.writer.add_histogram(f"{self.step_type}_{key}_hist", val, idx)


class ScalarMeanLogger(GenericScalarLogger):
    def __init__(self, metric_keys: Iterable[str], **kwargs):
        super(ScalarMeanLogger, self).__init__(metric_keys, torch.mean, **kwargs)


class ScalarMaxLogger(GenericScalarLogger):
    def __init__(self, metric_keys: Iterable[str], **kwargs):
        super(ScalarMaxLogger, self).__init__(metric_keys, torch.max, **kwargs)


class GenericWorstCaseLogger(LightingLoggingHandler):
    def __init__(
        self,
        metric_key,
        recording_keys: List[Tuple[str, Callable]],
        relevant_record_accessor,
        num_samples=5,
        **kwargs,
    ):
        """
        recording_keys: a list of [str, Callable]
            The Callable is used to flat tensor according to relevant_agents, non-relevant_agents will be removed.

            def Callable(tensor/list, relevant_agents):
                # Aggregate tensor/list by relevant agents


        """
        super(GenericWorstCaseLogger, self).__init__(**kwargs)
        self.logger_name = metric_key
        self.metric_key = metric_key
        self.num_samples = num_samples
        self.recording_keys = recording_keys
        self.relevant_record_accessor = relevant_record_accessor
        # Store the values of the metric
        self.values = torch.Tensor()
        self.records = None

    def keep_max_values(self, value: torch.Tensor, records: dict):
        assert max(value.shape) == value.reshape(-1).shape[0], f"The metrics must be 1d, got {value.shape}"
        sorted_vals, indices = torch.sort(value, descending=True)
        indices_dev_set = {indices.device: indices}
        num_samples = min(self.num_samples, len(sorted_vals))

        # Record the other records, assume the 1st dim is the batch dim
        new_records = {}
        for k, record in records.items():
            if isinstance(record, list):
                new_records[k] = [record[indices[i]] for i in range(num_samples)]
            elif isinstance(record, torch.Tensor):
                record_device = record.device
                if record_device not in indices_dev_set:
                    indices_dev_set[record_device] = indices.to(record_device)
                indices_dev = indices_dev_set[record_device]
                new_records[k] = torch.index_select(record, 0, indices_dev[:num_samples]).detach().cpu()
            else:
                raise RuntimeError(f'Unrecognized sequence type: "{k}": {record}')

        return sorted_vals[:num_samples], indices, new_records

    def collect_max_value(self, data_dictionary: dict, relevant_record_index):
        value = data_dictionary[self.metric_key]
        records = {}
        for k, accessor in self.recording_keys:
            if k not in data_dictionary or data_dictionary[k] is None:
                continue
            data = accessor(data_dictionary[k], relevant_record_index)
            records[k] = data

        sorted_values, indices, records = self.keep_max_values(value, records)
        return sorted_values, indices, records

    def collate_value_records(self, value, records):
        values = torch.cat(value, dim=0)

        list_records = {k: [item for l in records[k] for item in l] for k in records if isinstance(records[k][0], list)}
        [records.pop(k) for k in list_records]

        tensor_records = {}
        for k in list(records.keys()):
            if ("predicted_trajectory" == k or "map_" in k) and isinstance(records[k][0], torch.Tensor):
                t = records.pop(k)
                # tensor can have different lengths, so pad
                if len(set([d.shape[1] for d in t])) > 1:
                    col_t = collate_fn_with_padding(k, t, 1)
                else:
                    col_t = collate_fn_with_padding(k, t, 0)

                tensor_records[k] = col_t.flatten(0, 1)

        other_tensors = {k: torch.cat(records[k], dim=0) for k in records if isinstance(records[k][0], torch.Tensor)}
        tensor_records.update(other_tensors)
        tensor_records.update(list_records)
        return values, tensor_records

    @abstractmethod
    def log_values(self, sorted_values, indices, sorted_records, data):
        """Log the sorted value
        The relevant records are sorted accordingly by the worst case value.
        """

    def epoch_start(self, dataloader_type: str, data):
        super().epoch_start(dataloader_type, data)
        self.values = torch.Tensor()
        self.records = None

    def iteration_update(self, step_type, batch: dict, other: dict) -> None:
        super().iteration_update(step_type, batch, other)

        relevant_record_index = self.relevant_record_accessor(batch)
        if self.metric_key in batch:
            values, _, records = self.collect_max_value(batch, relevant_record_index)

            values = values.cpu()

            if self.records is None:
                self.records = records
                self.values = values
            else:
                values = [self.values, values]

                for k in records:
                    records[k] = [self.records[k], records[k]]

                # Consolidate worst cases
                collate_val, collate_records = self.collate_value_records(values, records)
                self.values, _, self.records = self.keep_max_values(collate_val, collate_records)

    def epoch_end(self, idx: int, global_batch_cnt: int, skip_visualization: bool) -> None:
        if len(self.values) == 0 or skip_visualization:
            return

        self.log_values(self.values, None, self.records, self.data)
