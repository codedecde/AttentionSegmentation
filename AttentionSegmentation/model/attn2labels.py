import re
from typing import List, Dict, Optional, Any
import os
import json
import pdb
from overrides import overrides
import logging
from copy import deepcopy

from allennlp.data.iterators import BasicIterator
from allennlp.data.instance import Instance
from allennlp.models.model import Model
from allennlp.common.tqdm import Tqdm
from nltk.corpus import stopwords
from nltk import word_tokenize

from AttentionSegmentation.visualization.visualize_attns import \
    colorized_predictions_to_webpage_binary, colorized_predictions_to_webpage
from AttentionSegmentation.evaluation.conlleval_perl import \
    fscore_from_preds
from AttentionSegmentation.reader.weak_conll_reader import NUM_TOKEN


logger = logging.getLogger(__name__)


def get_nltk_stopwords() -> List[str]:
    """Gets the NLTK stopwords + the tokenized version of the stopword

    ..note::
        NLTK stopwords have words like it's. However if you tokenize it,
        with say nltk tokenizer, it splits the words into it and 's. This
        is also observed in the CoNLL corpus. In order for the 's to be identified
        as a stopword, we tokenize the word and add all the tokenized components.

    """
    _stopword_list: List[str] = stopwords.words("english")
    stopword_list: List[str] = []
    for word in _stopword_list:
        stopword_list.append(word)
        for w in word_tokenize(word):
            stopword_list.append(w)
    stopword_list = list(set(stopword_list))
    return stopword_list


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

    def _get_filtered_set(self):
        """
        The set of words/symbols to be filtered out
        """
        return set()

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
                * pred (List[Tuple[str, float]]): The predicted labels and
                    probs. Can potentially have multiple labels being
                    predicted
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
        matrix = {
            self._indexer.ix2tags[ix]: {
                "tp": 0., "fp": 0, "fn": 0., "tn": 0.
            } for ix in range(len(self._indexer.ix2tags))
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
                    preds_probs=pred,
                    attns=attn
                )
                assert len(pred_labels) == len(gold_labels) == len(text)
                gold_set = set(gold)
                pred_set, _ = [set(list(x)) for x in zip(*pred)]
                # import pdb; pdb.set_trace()
                for tag in matrix:
                    if tag in gold_set and tag in pred_set:
                        matrix[tag]["tp"] += 1
                    elif tag not in gold_set and tag in pred_set:
                        matrix[tag]["fp"] += 1
                    elif tag in gold_set and tag not in pred_set:
                        matrix[tag]["fn"] += 1.
                    else:
                        matrix[tag]["tn"] += 1.
                preds = [
                    [x[0], float(x[1])] for x in pred
                ]
                prediction = {
                    "text": text,
                    "pred": preds,
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
            accs = []
            for tag in matrix:
                acc = (matrix[tag]["tp"] + matrix[tag]["tn"]) / \
                    sum(matrix[tag].values()) * 100.
                logger.info(f"Tag: {tag}, Acc: {acc:.2f}")
                accs.append(acc)
            avg_acc = sum(accs) / len(accs)
            logger.info(f"Average ACC: {avg_acc:.2f}")
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
                if attns[self._tag][ix] < self._tol or \
                        text[ix].lower() in self._get_filtered_set():
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
            predictions, visualization_filename
        )

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
                 tol=0.01, use_probs=False):
        super(BasicMultiPredictions, self).__init__(
            vocab, reader, visualize)
        self._tol = tol
        self._use_probs = use_probs

    @overrides
    def get_segmentation_from_prediction(
        self, preds_probs, attns, text, **kwargs
    ):
        attns = deepcopy(attns)
        preds = []
        lbl_probs = {}
        for pred, lbl_prob in preds_probs:
            preds.append(pred)
            if pred != "O":
                lbl_probs[pred] = lbl_prob
        if self._use_probs:
            for lbl in lbl_probs:
                for ix in range(len(attns[lbl])):
                    attns[lbl][ix] *= lbl_probs[lbl]
        if "O" in preds:
            assert len(preds) == 1
            pred_labels = ["O" for _ in range(len(text))]
        else:
            pred_labels = []
            for ix in range(len(text)):
                prob, tag = max([(attns[tag][ix], tag) for tag in attns
                                 if tag in preds])
                if prob < self._tol or \
                        text[ix].lower() in self._get_filtered_set() or NUM_TOKEN in text[ix].lower():
                    pred_labels.append("O")
                else:
                    if len(pred_labels) > 0 and re.match(
                            f".-{tag}", pred_labels[ix - 1]):
                        pred_labels.append(f"I-{tag}")
                    else:
                        pred_labels.append(f"B-{tag}")
        return pred_labels

    @overrides
    def visualize(self, predictions, visualization_filename):
        colorized_predictions_to_webpage(
            predictions, visualization_filename
        )

    @classmethod
    def from_params(cls, vocab, reader, params):
        tol = params.pop("tol", 0.01)
        visualize = params.pop("visualize", False)
        use_probs = params.pop("use_probs", False)
        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   reader=reader,
                   visualize=visualize, tol=tol, use_probs=use_probs)


class SymbolFilteredBinaryPredictions(BasicBinaryPredictions):

    def __init__(self,
                 vocab, reader,
                 visualize=False,
                 tol=0.01):
        super(SymbolFilteredBinaryPredictions, self).__init__(
            vocab, reader, visualize, tol)
        self._punct_set = set([".", ",", "!", "-", "?", "'", ")", "("])

    @overrides
    def _get_filtered_set(self):
        return self._punct_set


class SymbolStopwordFilteredBinaryPredictions(BasicBinaryPredictions):

    def __init__(self,
                 vocab, reader,
                 visualize=False,
                 tol=0.01):
        super(SymbolStopwordFilteredBinaryPredictions, self).__init__(
            vocab, reader, visualize, tol)
        stopword_list = get_nltk_stopwords()
        stop_set = set(stopword_list)
        punct_set = set([".", ",", "!", "-", "?", "'", ")", "(", "'s"])
        self._filter_set = stop_set | punct_set

    @overrides
    def _get_filtered_set(self):
        return self._filter_set


class SymbolFilteredMultiPredictions(BasicMultiPredictions):

    def __init__(self,
                 vocab, reader,
                 visualize=False,
                 tol=0.01,
                 use_probs=False):
        super(SymbolFilteredMultiPredictions, self).__init__(
            vocab, reader, visualize, tol, use_probs)
        self._punct_set = set([".", ",", "!", "-", "?", "'", ")", "("])

    @overrides
    def _get_filtered_set(self):
        return self._punct_set


class SymbolStopwordFilteredMultiPredictions(BasicMultiPredictions):

    def __init__(self,
                 vocab, reader,
                 visualize=False,
                 tol=0.01,
                 use_probs=False):
        super(SymbolStopwordFilteredMultiPredictions, self).__init__(
            vocab=vocab,
            reader=reader,
            visualize=visualize,
            tol=tol,
            use_probs=use_probs
        )
        stopword_list = get_nltk_stopwords()
        stop_set = set(stopword_list)
        punct_set = set([".", ",", "!", "-", "?", "'", ")", "("])
        self._filter_set = stop_set | punct_set

    @overrides
    def _get_filtered_set(self):
        return self._filter_set
