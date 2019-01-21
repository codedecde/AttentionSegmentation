from __future__ import absolute_import
from collections import OrderedDict
from typing import List
from overrides import overrides
import numpy as np
import pdb

from allennlp.data.tokenizers import Token

from visualization.visualize_attns \
    import plot_hierarchical_attn

# FIXME: Implement this
from preprocess.label_indices \
    import LabelIndicesBiMap, CondensedLabelIndicesBiMap


class BasePredictedInstance(object):
    """The base class.

    All PredictedInstance classes inherit from this

    """

    def as_dict(self):
        raise NotImplementedError("Child Class Implements this")

    def plot(self):
        raise NotImplementedError("Child Class Implements this")


class AttentionClassifierPredictedInstance(BasePredictedInstance):

    def __init__(self, tokens, mask,
                 attn_weights, pred_labels, gold_label=None, probs=None):
        super(AttentionClassifierPredictedInstance, self).__init__()
        assert len(tokens) == mask.sum(),\
            "Error. Expected {0}, found {1}".format(len(tokens), mask.sum())
        self._token_attentions = []
        self._predictions = pred_labels
        self._golds = gold_label
        self._probs = probs
        for ix in range(len(tokens)):
            token = tokens[ix]
            token_attentions = OrderedDict()
            for jx in range(attn_weights.shape[0]):
                _lbl = LabelIndicesBiMap.lookup(jx)
                attn = attn_weights[jx, ix]
                token_attentions[_lbl] = attn
            self._token_attentions.append((token, token_attentions))

    def get_attentions(self, key):
        retval = []
        for token, token_map in self._token_attentions:
            retval.append((token, token_map[key]))
        return retval


class HANPredictedInstance(BasePredictedInstance):
    """Class for storing predictions from the HAN model

    Arguments:
        sents (List[List[Token]]): The sentences
        sent_attn (``np.ndarray``): 1 x num_sents
            The sentence attentions
        word_attns (``np.ndarray``): num_sents x max_sent_len
            The (masked) word attentions
        pred_labels (List[str]): The predicted labels
        gold_labels (List[str]): The gold labels
        probs (List[float]): The probability of each class,
            predicted by the model
        tol (float): The tolerance for check. If instantiating
            it from a dict, then set a higher threshold

    """

    def __init__(
        self,
        sents: List[List[Token]],
        sent_attn: np.ndarray,
        word_attns: np.ndarray,
        pred_labels: List[str],
        gold_labels: List[str],
        probs: List[float],
        tol: float = 1e-3
    ) -> None:

        # A bunch of assertions
        super(HANPredictedInstance, self).__init__()
        num_sentences = len(sents)
        max_sent_len = max(list(map(len, sents)))
        try:
            assert (num_sentences, max_sent_len) == word_attns.shape
        except Exception as e:
            pdb.set_trace()
        self.sents = [[word.text for word in sent] for sent in sents]
        sent_attn_array = sent_attn[:len(sents)]
        self.sent_attn = self.convert_to_list(sent_attn_array)
        assert len(sents) == len(self.sent_attn)
        self.word_attns = []
        assert abs(1. - sent_attn_array.sum()) < tol
        for ix in range(len(sents)):
            try:
                word_attn_array = word_attns[ix].flatten()[:len(sents[ix])]
            except Exception as e:
                pdb.set_trace()
            word_attn = self.convert_to_list(word_attn_array)
            try:
                assert abs(1. - word_attn_array.sum()) < tol
            except Exception as e:
                pdb.set_trace()
            self.word_attns.append(word_attn)
        self.pred_labels = pred_labels or []
        self.gold_labels = gold_labels or []
        self.probs = probs

    @classmethod
    def from_dict(cls, json_obj):
        """Instantiate class from json object
        """
        sents = [[Token(x) for x in sent] for sent in json_obj["sentences"]]
        sent_attn = np.array(json_obj["sent_attn"])
        word_attn_list = json_obj["word_attns"]
        max_sent_len = max(list(map(len, word_attn_list)))
        word_attns = np.zeros((len(word_attn_list), max_sent_len))
        for ix in range(word_attns.shape[0]):
            for jx in range(len(word_attn_list[ix])):
                word_attns[ix, jx] = word_attn_list[ix][jx]
        return cls(
            sents=sents,
            sent_attn=sent_attn,
            word_attns=word_attns,
            pred_labels=json_obj["preds"],
            gold_labels=json_obj["golds"],
            probs=json_obj["label_probs"],
            tol=1e-2
        )

    @overrides
    def as_dict(self):
        """Return the class params in an OrderedDict

        Useful for dumping in a json

        """
        retval = OrderedDict()
        retval["sentences"] = self.sents
        retval["sent_attn"] = self.sent_attn
        retval["word_attns"] = self.word_attns
        retval["preds"] = self.pred_labels
        retval["golds"] = self.gold_labels
        retval["label_probs"] = self.probs
        return retval

    def convert_to_list(self, array: np.ndarray) -> List[float]:
        """Converts an ndarray into a (pretty) list of floats
        """
        return [float(f"{x:2.4f}") for x in array]

    @overrides
    def plot(self, filename=None):
        """Function for plotting the Instance

        Arguments:
            filename (Optional[str]): Save image
                to location.
        """
        data_point = self.as_dict()
        plot_hierarchical_attn(**data_point, filename=filename, fontsize=12)


def query_data_for_labels(
    data: List[BasePredictedInstance],
    pred_label: str,
    gold_label: str
):
    assert pred_label != "" or gold_label != "", \
        "Both preds and golds cannot be empty"
    if gold_label == "" or pred_label == "":
        # We are checking for empty
        retvals = []
        for ix, dp in enumerate(data):
            pred_set = set(dp.pred_labels)
            gold_set = set(dp.gold_labels)
            correct = pred_set & gold_set
            incorrect_gold = gold_set - correct
            incorrect_pred = pred_set - correct
            if pred_label == "":
                if gold_label in incorrect_gold and len(incorrect_pred) == 0:
                    retvals.append((ix, dp))
            if gold_label == "":
                if pred_label in incorrect_pred and len(incorrect_gold) == 0:
                    retvals.append((ix, dp))
        return retvals
    elif pred_label == gold_label:
        return [
            (ix, dp) for ix, dp in enumerate(data)
            if pred_label in dp.pred_labels and pred_label in dp.gold_labels
        ]
    else:
        return [
            (ix, dp) for ix, dp in enumerate(data)
            if (pred_label in dp.pred_labels and
                pred_label not in dp.gold_labels and
                gold_label in dp.gold_labels)
        ]
