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

from AttentionSegmentation.allennlp.common.tqdm import Tqdm
from AttentionSegmentation.allennlp.common.params import Params
from AttentionSegmentation.allennlp.data.fields \
    import TextField, MultiLabelField
from AttentionSegmentation.allennlp.nn import util
from AttentionSegmentation.allennlp.data.instance import Instance
from AttentionSegmentation.allennlp.common.checks import ConfigurationError
from AttentionSegmentation.allennlp.data.tokenizers import Token
from AttentionSegmentation.allennlp.data import Vocabulary
from AttentionSegmentation.allennlp.data.iterators import BasicIterator
from AttentionSegmentation.allennlp.data.iterators import DataIterator
from AttentionSegmentation.allennlp.data.token_indexers \
    import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer

from AttentionSegmentation.visualization.visualize_attns \
    import colorized_list_to_webpage, colorized_predictions_to_webpage

from AttentionSegmentation.commons.utils import \
    read_from_config_file, to_numpy
import AttentionSegmentation.model.classifiers as Models
import AttentionSegmentation.reader as Readers

# from Model.predicted_instances\
#     import HANPredictedInstance, AttentionClassifierPredictedInstance
# from Model.metrics import ConfusionMatrix


logger = logging.getLogger(__name__)


class BaseModelRunner(object):
    """An Abstract class, providinf a common
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

        text_field_embedder = model_params.get("text_field_embedder", None)
        if text_field_embedder is not None:
            tokens = text_field_embedder.get("tokens", None)
            if tokens is not None:
                tokens.pop("pretrained_file", None)
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
        self._model.load_state_dict(model_state)
        logger.info("Model Loaded")
        self._num_labels = self._reader.get_label_indexer().get_num_tags()

        # Build Iterator
        logger.info("Loading iterator")
        # Use a basic iterator
        # since bucket iterator changes the ordering of data
        # even when shuffle = False
        self._data_iterator = BasicIterator(batch_size=32)
        self._data_iterator.index_with(self._vocab)
        logger.info("iterator loaded")
        trainer_params = self._config.pop("trainer", None)
        self._cuda_device = -1
        if trainer_params is not None and torch.cuda.is_available():
            self._cuda_device = trainer_params.pop_int("cuda_device", -1)
        if self._cuda_device != -1:
            self._model.cuda(self._cuda_device)

    def _get_text_from_instance(self, instance: Instance) -> List[str]:
        """Helper function to extract text from an instance
        """
        return list(map(lambda x: x.text, instance.fields['tokens'].tokens))

    def _process_file(self, filename: str):
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
    def _process_file(self, filename: str, html_file: str = ""):
        """
            Generates predictions from a file
        """
        instances = self._reader.read(filename)
        iterator = self._data_iterator(
            instances,
            num_epochs=1,
            shuffle=False,
            cuda_device=self._cuda_device,
            for_training=False
        )
        self._model.eval()
        num_batches = self._data_iterator.get_num_batches(instances)
        inference_generator_tqdm = Tqdm.tqdm(iterator, total=num_batches)
        predictions = []
        index = 0
        index_labeler = self._reader.get_label_indexer()
        sentences = []
        attentions = []
        correct_counts = 0.
        for batch in inference_generator_tqdm:
            # Currently I don't support multi-gpu data parallel
            output_dict = self._model.decode(self._model(**batch))
            for ix in range(len(output_dict["preds"])):
                text = self._get_text_from_instance(instances[index])
                label_num = instances[index].fields['labels'].labels[0]
                # FIXME: Currently supporting binary classification
                assert len(instances[index].fields['labels'].labels) == 1
                index += 1
                pred = output_dict["preds"][ix]
                attn = output_dict["attentions"][ix]
                gold = "O"
                if html_file != "":
                    sentences.append(" ".join(text))
                    attentions.append(attn)
                if label_num < len(index_labeler.ix2tags):
                    gold = index_labeler.ix2tags[label_num]
                if pred == gold:
                    correct_counts += 1.
                prediction = {
                    "text": text,
                    "pred": pred,
                    "attn": attn,
                    "gold": gold
                }
                predictions.append(prediction)
        if html_file != "":
            # colorized_list_to_webpage(
            #     sentences, attentions, vis_page=html_file)
            colorized_predictions_to_webpage(
                predictions, vis_page=html_file)
        print("Accuracy: ", 100 * correct_counts / index)
        return predictions


if __name__ == "__main__":
    base_dir = "./Experiments/CoNLL/HAN-ELMO-Experiments/run-20/"
    runner = BasicAttentionModelRunner.load_from_dir(base_dir)
    base_valid_dir = "./Data/CoNLLData/"
    valid_file = f"{base_valid_dir}/valid.txt"
    runner._process_file(valid_file, "./WebOuts/visualize.html")
