import re
from typing import List, Dict, Optional, Any
import os
import json
import pdb
from overrides import overrides
import logging

from allennlp.data.iterators import BasicIterator
from allennlp.data.instance import Instance
from allennlp.models.model import Model
from allennlp.common.tqdm import Tqdm

from AttentionSegmentation.visualization.visualize_attns import \
    colorized_predictions_to_webpage_binary
from AttentionSegmentation.evaluation.conlleval_perl import \
    fscore_from_preds


logger = logging.getLogger(__name__)


def get_binary_preds_from_attns(attns, tag, tol=0.01):
    tag_list = []
    for ix in range(len(attns)):
        if attns[ix] < tol:
            tag_list.append("O")
        else:
            if len(tag_list) > 0 and re.match(f".*-{tag}", tag_list[ix - 1]):
                tag_list.append(f"I-{tag}")
            else:
                tag_list.append(f"B-{tag}")
    return tag_list


class BasePredictionClass(object):
    """
    This (Abstract) class is devoted to extracting predictions,
    managing storage of predictions, and the optional
    visualization of predictions
    """

    def __init__(self, vocab, reader, visualize=False):
        self._vocab = vocab
        self._iterator = BasicIterator(batch_size=32)
        self._iterator.index_with(self._vocab)
        self._reader = reader
        self._indexer = self._reader.get_label_indexer()
        self._visualize = visualize

    def _get_text_from_instance(self, instance: Instance) -> List[str]:
        """Helper function to extract text from an instance
        """
        return list(map(lambda x: x.text, instance.fields['tokens'].tokens))

    def get_segmentation_from_prediction(
        self,
        *args, **kwargs
    ) -> List[str]:
        raise NotImplementedError("The child class implements this")

    def visualize(self, *args, **kwargs):
        raise NotImplementedError("The child class implements this")

    def get_predictions(self, instances: List[Instance], model: Model,
                        cuda_device: int = -1,
                        prediction_file: Optional[str] = None,
                        visualization_file: Optional[str] = None,
                        verbose: bool = False
                        ) -> List[Dict]:
        """
        We use this function to get predictions
        We use a basic itereator, since a bucket iterator shuffles
        data, even for shuffle=False

        Arguments:
            data (List[Instance]) : The list of instances for inference
            model (Model) : The model being used for predictions
            cuda_device (int) : The cuda device being used for processing
            verbose (bool) : Log accuracies and such

        Returns:
            predictions (List[Dict]) : The predictions. Each contains the
                following keys
                * text (List[str]): The tokens
                * pred (List[str]): The predicted labels
                    can potentially have multiple labels being predicted
                * gold (List[str]): The gold labels
                    can potentially have multiple gold labels
                * pred_labels (List[str]): Predicted labels for segmentation
                    Note that an this method is implemented by the base classes
                * attn (Dict[str, List[float]]) : A dictionary mapping tags to
                    attention values
                * gold_labels : The gold labels for segmentation
                    The gold labels for segmentation

        Additionally, this class stores the base_predictions, as well as the
            visualization, if visualization is set to True, and base_dir is
             provided
        """
        iterator = self._iterator(
            instances,
            num_epochs=1,
            shuffle=False,
            cuda_device=cuda_device,
            for_training=False
        )
        model.eval()
        num_batches = self._iterator.get_num_batches(instances)
        inference_generator_tqdm = Tqdm.tqdm(iterator, total=num_batches)
        predictions = []
        index = 0
        correct_counts = {
            self._indexer.ix2tags[ix]: 0 for ix in range(len(
                self._indexer.ix2tags))
        }
        all_counts = {
            self._indexer.ix2tags[ix]: 0 for ix in range(len(
                self._indexer.ix2tags))
        }
        for batch in inference_generator_tqdm:
            # Currently I don't support multi-gpu data parallel
            output_dict = model.decode(model(**batch))
            for ix in range(len(output_dict["preds"])):
                text = self._get_text_from_instance(instances[index])
                pred = output_dict["preds"][ix]
                gold = [self._indexer.get_tag(label)
                        for label in instances[index].fields['labels'].labels
                        ]
                attn = output_dict["attentions"][ix]
                gold_labels = instances[index].fields['tags'].labels
                assert all([len(attn[x]) == len(text) for x in attn])
                gold_labels = self._indexer.extract_relevant(gold_labels)
                pred_labels = self.get_segmentation_from_prediction(
                    text=text,
                    preds=pred,
                    attns=attn
                )
                assert len(pred_labels) == len(gold_labels) == len(text)
                for t in all_counts:
                    all_counts[t] += 1.

                for t in correct_counts:
                    if t not in pred and t in gold or \
                            t in pred and t not in gold:
                        continue
                    correct_counts[t] += 1.

                prediction = {
                    "text": text,
                    "pred": pred,
                    "gold": gold,
                    "attn": attn,
                    "pred_labels": pred_labels,
                    "gold_labels": gold_labels
                }
                predictions.append(prediction)
                index += 1
        if prediction_file is not None and prediction_file != "":
            with open(prediction_file, "w") as f:
                json.dump(predictions, f, ensure_ascii=True, indent=4)
        if visualization_file is not None and self._visualize and \
                visualization_file != "":
            self.visualize(predictions, visualization_file)
        if verbose:
            for p in all_counts:
                acc = correct_counts[p] / all_counts[p] * 100.
                logger.info(f"Tag: {p}, Acc: {acc:.2f}")
            p, r, f = fscore_from_preds(predictions, False)
        return predictions


class BasicBinaryPredictions(BasePredictionClass):

    def __init__(self,
                 vocab, reader,
                 visualize=False,
                 tol=0.01):
        super(BasicBinaryPredictions, self).__init__(
            vocab, reader, visualize)
        indexer = reader.get_label_indexer()
        self._tag = indexer.get_tag(0)
        self._tol = tol

    @overrides
    def get_segmentation_from_prediction(
        self, preds, attns, text, **kwargs
    ):
        assert len(preds) == 1
        tag = preds[0]
        if tag == "O":
            pred_labels = ["O" for _ in range(len(text))]
        else:
            pred_labels = []
            for ix in range(len(attns[self._tag])):
                if attns[self._tag][ix] < self._tol:
                    pred_labels.append("O")
                else:
                    if len(pred_labels) > 0 and re.match(
                            f".*-{tag}", pred_labels[ix - 1]):
                        pred_labels.append(f"I-{tag}")
                    else:
                        pred_labels.append(f"B-{tag}")
        return pred_labels

    @overrides
    def visualize(self, predictions, visualization_filename):
        colorized_predictions_to_webpage_binary(
            predictions, visualization_filename)

    @classmethod
    def from_params(cls, vocab, reader, params):
        tol = params.pop("tol", 0.01)
        visualize = params.pop("visualize", False)
        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   reader=reader,
                   visualize=visualize, tol=tol)


class BasicMultiPredictions(BasePredictionClass):
    """This class deals with multi tag classification
    """

    def __init__(self,
                 vocab, reader,
                 visualize=False,
                 tol=0.01, use_prod=False):
        super(BasicMultiPredictions, self).__init__(
            vocab, reader, visualize)
        self._tol = tol
        self._use_prod = use_prod

    @overrides
    def get_segmentation_from_prediction(
        self, preds, attns, text, **kwargs
    ):
        if "O" in preds:
            assert len(preds) == 1
            pred_labels = ["O" for _ in range(len(text))]
        else:
            pred_labels = []
            for ix in range(len(text)):
                prob, tag = max([(attns[tag][ix], tag) for tag in attns])
                if prob < self._tol:
                    pred_labels.append("O")
                else:
                    if len(pred_labels) > 0 and re.match(
                            f".-{tag}", pred_labels[ix - 1]):
                        pred_labels.append(f"I-{tag}")
                    else:
                        pred_labels.append(f"B-{tag}")
        return pred_labels

    @classmethod
    def from_params(cls, vocab, reader, params):
        tol = params.pop("tol", 0.01)
        visualize = params.pop("visualize", False)
        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   reader=reader,
                   visualize=visualize, tol=tol)
