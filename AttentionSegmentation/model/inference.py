from __future__ import absolute_import
from typing import List, Dict, Union, Optional, Tuple, MutableSet, Any
import torch
import logging
import io
from overrides import overrides
import json
from collections import OrderedDict
import os
import numpy as np
import json
import pdb
import re

from allennlp.common.tqdm import Tqdm
from allennlp.common.params import Params
from allennlp.models.model import remove_pretrained_embedding_params
from allennlp.data.fields \
    import TextField, MultiLabelField
from allennlp.nn import util
from allennlp.data.instance import Instance
from allennlp.common.checks import ConfigurationError
from allennlp.data.tokenizers import Token
from allennlp.data import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.data.iterators import DataIterator
from allennlp.data.token_indexers \
    import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer

from AttentionSegmentation.visualization.visualize_attns \
    import colorized_predictions_to_webpage

from AttentionSegmentation.commons.utils import \
    read_from_config_file, to_numpy
import AttentionSegmentation.model.classifiers as Models
from AttentionSegmentation.model.attn2labels import \
    get_binary_preds_from_attns
import AttentionSegmentation.reader as Readers
import AttentionSegmentation.model.attn2labels as SegmentationModels

# from Model.predicted_instances\
#     import HANPredictedInstance, AttentionClassifierPredictedInstance
# from Model.metrics import ConfusionMatrix


logger = logging.getLogger(__name__)


class BaseModelRunner(object):
    """An Abstract class, providing a common
    way of interfacing with the Classifiers.

    Implements commonly used functionalities, like
    loading models and such.
    """

    def __init__(
            self, config_file: str, model_path: str,
            vocab_dir: str, base_embed_dir: Optional[str] = None):
        self._config = read_from_config_file(config_file)
        # Load Vocab
        logger.info("Loading Vocab")
        self._vocab = Vocabulary.from_files(vocab_dir)
        logger.info("Vocab Loaded")

        # Load Reader
        dataset_reader_params = self._config.pop("dataset_reader")
        reader_type = dataset_reader_params.pop("type", None)
        assert reader_type is not None and hasattr(Readers, reader_type),\
            f"Cannot find reader {reader_type}"
        reader = getattr(Readers, reader_type).from_params(
            dataset_reader_params)
        self._reader = reader
        # Load Model
        model_params = self._config.pop("model")
        model_type = model_params.pop("type")

        if base_embed_dir is not None:
            # This hack is necessary to ensure we load ELMO embeddings
            # correctly. Whoever discovers this later,
            # this is ugly. I apologise
            text_field_params = model_params.get("text_field_embedder", None)
            if text_field_params is not None:
                elmo_params = text_field_params.get("elmo", None)
                if elmo_params is not None:
                    options_file_path = elmo_params.get("options_file")
                    weight_file_path = elmo_params.get("weight_file")
                    _, options_file = os.path.split(options_file_path)
                    _, weight_file = os.path.split(weight_file_path)
                    complete_options_file = os.path.join(
                        base_embed_dir, options_file)
                    complete_weight_file = os.path.join(
                        base_embed_dir, weight_file)
                    elmo_params["options_file"] = complete_options_file
                    elmo_params["weight_file"] = complete_weight_file

        remove_pretrained_embedding_params(
            model_params
        )
        assert model_type is not None and hasattr(Models, model_type),\
            f"Cannot find reader {model_type}"
        self._model = getattr(Models, model_type).from_params(
            vocab=self._vocab,
            params=model_params,
            label_indexer=reader.get_label_indexer()
        )
        logger.info("Loading Model")
        model_state = torch.load(model_path,
                                 map_location=util.device_mapping(-1))
        old_keys = []
        new_keys = []
        for key in model_state.keys():
            new_key = None
            if "gamma" in key:
                # FIXME : Make this regex based
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        # pdb.set_trace()
        # for old_key, new_key in zip(old_keys, new_keys):
        #     model_state[new_key] = model_state.pop(old_key)
        # pdb.set_trace()
        self._model.load_state_dict(model_state)
        logger.info("Model Loaded")
        self._num_labels = self._reader.get_label_indexer().get_num_tags()
        segmenter_params = self._config.pop("segmentation")
        segment_class = segmenter_params.pop("type")
        self._segmenter = getattr(
            SegmentationModels, segment_class).from_params(
                vocab=self._vocab,
                reader=reader,
                params=segmenter_params
        )
        trainer_params = self._config.pop("trainer", None)
        self._cuda_device = -1
        if trainer_params is not None and torch.cuda.is_available():
            self._cuda_device = trainer_params.pop_int("cuda_device", -1)
        if self._cuda_device != -1:
            self._model.cuda(self._cuda_device)

    def generate_preds_from_file(self, filename: str):
        raise NotImplementedError("Child class has to implement this")

    @classmethod
    def load_from_dir(cls, base_dir, base_embed_dir=None):
        """Instantiates a ModelRunner from an experiment directory
        """
        config_file = os.path.join(base_dir, "config.json")
        assert os.path.exists(config_file),\
            f"Cannot find config file in {base_dir}"
        vocab_dir = os.path.join(base_dir, "vocab")
        assert os.path.exists(vocab_dir),\
            f"Cannot find vocab dir in {base_dir}"
        model_path = os.path.join(base_dir, "models", "best.th")
        assert os.path.exists(model_path),\
            f"Cannot find Best model in {base_dir}"
        return cls(
            config_file=config_file,
            vocab_dir=vocab_dir,
            model_path=model_path,
            base_embed_dir=base_embed_dir
        )


class BasicAttentionModelRunner(BaseModelRunner):
    """ This provides an interface for the Classifier
    (AttentionSegmentation.model.classifiers)

    Arguments:
        config_file (str): Config file used to create model
        model_path (str): Path to load model from
        vocab_dir (str): Path to the vocab dir
        base_embed_dir (Optional[str]):
            The base directory for embeddings

    """

    def __init__(self,
                 config_file: str,
                 model_path: str,
                 vocab_dir: str,
                 base_embed_dir: Optional[str] = None):
        super(BasicAttentionModelRunner, self).__init__(
            config_file=config_file, model_path=model_path,
            vocab_dir=vocab_dir, base_embed_dir=base_embed_dir)

    @overrides
    def generate_preds_from_file(
        self,
        filename: str, prediction_file: str = "",
        visualization_file: str = ""):
        """
            Generates predictions from a file
        """
        instances = self._reader.read(filename)
        self._model.eval()
        predictions = self._segmenter.get_predictions(
            instances=instances,
            model=self._model,
            cuda_device=self._cuda_device,
            prediction_file=prediction_file,
            visualization_file=visualization_file,
            verbose=True)
        return predictions


if __name__ == "__main__":
    pass
