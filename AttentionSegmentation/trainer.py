import os
import torch
from collections import defaultdict
from typing import Dict, Tuple, Any
from overrides import overrides
import numpy as np
import logging
import re
import pdb

from allennlp.nn import util
from allennlp.common.tqdm import Tqdm
import AttentionSegmentation.commons.trainer as common_trainer
from AttentionSegmentation.visualization.tensorboard_logger import TfLogger

logger = logging.getLogger(__name__)


def get_loss_string(name, value):
    """Returns string with name and value of different losses
    """
    retval = ""
    if value < 1e-3:
        retval = f"{name}: {value:2.2e}"
    else:
        retval = f"{name}: {value:2.2f}"
    return retval


class Trainer(common_trainer.Trainer):
    def __init__(self, base_dir, tensorboard=None, *args, **kwargs):
        # Setup Tensorboard Logging
        if tensorboard is not None:
            self._tf_logger, self._tf_params = self._init_tensorboard(
                tensorboard, base_dir)
        else:
            self._tf_logger = None
            self._tf_params = None
        super(Trainer, self).__init__(base_dir=base_dir, *args, **kwargs)
        self._reset_counter()

    def _init_tensorboard(
        self, tensorboard, base_dir
    ):
        tf_params = tensorboard
        tf_logger = TfLogger(base_dir, "tf_logging")
        return tf_logger, tf_params

    def _tf_log(self, metrics, step):
        for metric in metrics:
            if metric in self._tf_params["log_summary"]:
                log_func = self._tf_params["log_summary"][metric]
                getattr(self._tf_logger, log_func)(
                    tag=metric, value=metrics[metric], step=step)

    def _reset_counter(self):
        self._zero_counts = {"zero": 0., "num_preds": 0}

    @overrides
    def train(self, *args, **kwargs):
        super(Trainer, self).train(*args, **kwargs)

    @overrides
    def test(self, data):
        """Test model, by loading best.th, and getting metrics
        on data
        """
        model_path = os.path.join(self._serialization_dir, "best.th")
        logger.info("Loading best model from {0}".format(model_path))
        model_state = torch.load(model_path,
                                 map_location=util.device_mapping(-1))
        self._model.load_state_dict(model_state)
        # FIXME: IMPLEMENT THIS
        # loss, num_batches = self._inference_loss(data, logger_string="Testing")
        # metrics = self._get_metrics(loss, num_batches, reset=True)
        # logger.info("Testing Results now")
        # writebuf = self._description_from_metrics(metrics)
        # logger.info(writebuf)
        # if self._segmenter is not None:
        #     prediction_file = os.path.join(
        #         self._base_dir, "test_predictions.json")
        #     visualization_file = os.path.join(
        #         self._base_dir, "visualization", "test.html")
        #     self._segmenter.get_predictions(
        #         instances=data,
        #         model=self._model,
        #         cuda_device=self._iterator_device,
        #         prediction_file=prediction_file,
        #         visualization_file=visualization_file,
        #         verbose=True)

    @classmethod
    @overrides
    def from_params(cls, *args, **kwargs) -> 'Trainer':
        # model_serialization_dir = os.path.join(base_dir, "models")
        # assert os.path.exists(model_serialization_dir), \
        #     "Cannot find the serialization directory at" \
        #     f" {model_serialization_dir}"
        # kwargs["serialization_dir"] = model_serialization_dir
        tensorboard = None
        if "tensorboard" in kwargs["params"]:
            tensorboard = kwargs["params"].pop("tensorboard")
        new_args = cls.get_args(*args, **kwargs)
        return Trainer(tensorboard=tensorboard, **new_args)
