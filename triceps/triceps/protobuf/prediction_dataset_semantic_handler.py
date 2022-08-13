import json
from typing import List

import numpy as np
from google.protobuf.timestamp_pb2 import Timestamp

import triceps.protobuf
from radutils.misc import parse_protobuf_timestamp
from triceps.protobuf.prediction_dataset import ProtobufPredictionDataset
from triceps.protobuf.prediction_dataset_auxiliary import InputsHandler

# These are the column indices in the semantic label tensor output from SemanticLabelsHandler.
# The tensor is (B x) Num_entries x 5, where Num_entries is some cap numbers to fix a tensor, and the 5 columns are:
# Boolean value - is this item in the tensor valid?
SEMANTIC_HANDLER_VALID_IDX = 0
# When is the start of the semantic interval?
SEMANTIC_HANDLER_START_IDX = 1
# When is the end of the semantic interval?
SEMANTIC_HANDLER_FINISH_IDX = 2
# What is the label type of the interval? (e.g. "intent to cross", "is stopped", "aware of egovehicle",..)
SEMANTIC_HANDLER_TYPE_IDX = 3
# What is the value of the interval (can be -1 or 1, but in theory can populate other values)
SEMANTIC_HANDLER_VALUE_IDX = 4

# Type ID for language tokens
TYPE_LANGUAGE_TOKENS = "language tokens"


class SemanticLabelsHandler(InputsHandler):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.definitions = None
        self.replace_ids = False
        if "latent_factors_file" in self.params and self.params["latent_factors_file"] is not None:
            with open(self.params["latent_factors_file"]) as fp:
                self.definitions = json.load(fp)
            if self.replace_ids:
                for i, itm in enumerate(self.definitions):
                    itm["id"] = i

    def _get_params_for_hash(self, params: dict) -> dict:
        """Override to remove file absolute path"""
        p = super()._get_params_for_hash(params)
        # Only keep the filename of the latent_factors_file
        p["latent_factors_file"] = params["latent_factors_file"].split("/")[-1]
        p = dict(sorted(p.items()))
        return p

    def get_hash_param_keys(self) -> List[str]:
        return [
            "use_semantics",
            "latent_factors_file",
            "max_semantic_targets",
        ]

    def _process_impl(
        self,
        result_dict: dict,
        params,
        filename,
        index,
    ):
        ret = {}
        # An array with valid,start,finish,type,value
        # The overall number of entries in the semantic tensor.
        num_semantic_targets = min(len(self.definitions), params["max_semantic_targets"])
        ret[ProtobufPredictionDataset.DATASET_KEY_SEMANTIC_LABELS] = np.zeros([num_semantic_targets, 5])
        if result_dict[ProtobufPredictionDataset.DATASET_KEY_SEMANTIC_TARGETS]:
            # Only select target types from self.definitions
            # change to: go over instance.semantic_targets, add entries to the table until cnt==num_semantic_targets
            for cnt, defn in enumerate(self.definitions):
                if cnt >= num_semantic_targets:
                    break
                type_numeric = 0.0
                value = 0.0
                is_valid = True
                matched_type = False
                timestamp_start = 0.0
                timestamp_end = 0.0
                # Search target from pb instance
                for tgt in result_dict[ProtobufPredictionDataset.DATASET_KEY_SEMANTIC_TARGETS]:
                    type_id = tgt["typeId"]
                    if defn["annotator_question"].lower() == type_id:
                        type_numeric = float(defn["id"])
                        matched_type = True
                        break
                is_valid = is_valid and matched_type
                if matched_type:
                    if "value" in tgt:
                        value = float(tgt["value"])
                    else:
                        is_valid = False
                    if "timestamp_start" in tgt:
                        timestamp_start = parse_protobuf_timestamp(tgt["timestamp_start"]).ToNanoseconds() / 1e9
                    if "timestamp_end" in tgt:
                        timestamp_end = parse_protobuf_timestamp(tgt["timestamp_end"]).ToNanoseconds() / 1e9
                ret[ProtobufPredictionDataset.DATASET_KEY_SEMANTIC_LABELS][cnt, :] = np.array(
                    [float(is_valid), timestamp_start, timestamp_end, type_numeric, value]
                )

        return ret
