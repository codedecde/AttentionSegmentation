from collections import defaultdict, OrderedDict
from typing import Dict
from torch import LongTensor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

from allennlp.training.metrics.metric import Metric

# FIXME: Implement this
from preprocess.label_indices \
    import LabelIndicesBiMap, CondensedLabelIndicesBiMap


class ClassificationMetrics(Metric):
    def __init__(self, num_labels, label_indexer):
        self.num_labels = num_labels
        self.label_indexer = eval(label_indexer)
        super(ClassificationMetrics, self).__init__()
        self.setUp()

    def setUp(self):
        self._counts = OrderedDict()
        for label in self.label_indexer.label2ix:
            ix = self.label_indexer.lookup(label)
            if ix == self.num_labels:
                break
            self._counts[ix] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

    def _update_counts(self, pred: float, gold: float, lbl: int):
        assert isinstance(pred, int)
        assert isinstance(gold, int)
        assert isinstance(lbl, int)
        if pred == gold:
            if pred == 1:
                self._counts[lbl]['tp'] += 1
            else:
                self._counts[lbl]['tn'] += 1
        else:
            if pred == 1:
                self._counts[lbl]['fp'] += 1
            else:
                self._counts[lbl]['fn'] += 1

    def __call__(self, pred_labels: LongTensor, gold_labels: LongTensor):
        """
            Arguments:
                pred_labels (``LongTensor``): batch x L
                    Boolean Array indicating predicting of label or not
                gold_labels (``LongTensor``): batch x L
                    Boolean Array indicating presence of label or not
        """
        pred_labels, gold_labels = self.unwrap_to_tensors(
            pred_labels, gold_labels)
        pred_labels, gold_labels = pred_labels.long(), gold_labels.long()
        batch_sz, num_labels = pred_labels.size()
        for bt in range(batch_sz):
            for lbl in range(num_labels):
                self._update_counts(
                    pred_labels[bt, lbl],
                    gold_labels[bt, lbl],
                    lbl
                )

    def _get_metrics(self, counts: Dict[str, int])->Dict[str, float]:
        metric = OrderedDict()
        tol = 1e-5
        metric["precision"] = counts["tp"] / (counts["tp"] + counts["fp"] + tol)
        metric["recall"] = counts["tp"] / (counts["tp"] + counts["fn"] + tol)
        metric["fscore"] = 2 * metric["precision"] * metric["recall"]
        metric["fscore"] /= (metric["precision"] + metric["recall"] + tol)
        metric["accuracy"] = (counts["tp"] + counts["tn"])
        metric["accuracy"] /= sum([counts[x] for x in counts])
        return metric

    def get_metric(self, reset: bool = False):
        metrics = OrderedDict()
        for ix in self._counts:
            label = self.label_indexer.lookup(ix)
            metrics[label] = self._get_metrics(self._counts[ix])
        if reset:
            self.setUp()
        return metrics


class ConfusionMatrix(object):
    """Computes the confusion matrix

    This class builds the confusion matrix for
    the predictive models

    Arguments:
        data (List[Dict[str, Any]]): The prediction data,
            as a dictionary
        label_indexer (``LabelIndicesBiMap``): The label indexer
            ("LabelIndicesBiMap|CondensedLabelIndicesBiMap")

    """

    def __init__(self):
        self.confusion_matrix = None
        self.labels = None

    def plot(self, filename=None):
        gold_labels = self.labels
        pred_labels = self.labels
        df_cm = pd.DataFrame(
            self.confusion_matrix, index=[i for i in gold_labels],
            columns=[i for i in pred_labels])
        plt.figure(figsize=(10, 7))
        ax = sn.heatmap(df_cm, annot=True, fmt="3.1f")
        ax.set(xlabel="Pred Labels", ylabel="Gold Labels")
        plt.title("Confusion Matrix for Classification")
        plt.tight_layout()

    def load_data(self, data, label_indexer):
        # FIXME: Implement this
        num_labels = label_indexer.get_num_labels()
        self.confusion_matrix = np.zeros((num_labels + 1, num_labels + 1))
        for dp in data:
            pred_set = set(dp["preds"])
            gold_set = set(dp["golds"])
            correct = pred_set & gold_set
            for p in correct:
                p_ix = label_indexer.lookup(p)
                self.confusion_matrix[p_ix, p_ix] += 1
            incorrect_pred = pred_set - correct
            incorrect_gold = gold_set - correct
            if len(incorrect_pred) == 0 and len(incorrect_gold) == 0:
                continue
            elif len(incorrect_pred) == 0:
                for g in incorrect_gold:
                    g_ix = label_indexer.lookup(g)
                    self.confusion_matrix[g_ix, num_labels] += 1
            elif len(incorrect_gold) == 0:
                for p in incorrect_pred:
                    p_ix = label_indexer.lookup(p)
                    self.confusion_matrix[num_labels, p_ix] += 1
            else:
                for p in incorrect_pred:
                    p_ix = label_indexer.lookup(p)
                    for g in incorrect_gold:
                        g_ix = label_indexer.lookup(g)
                        self.confusion_matrix[g_ix, p_ix] += 1
        self.labels = [label_indexer.lookup(ix)
                       for ix in range(num_labels)] + ["Empty"]

    def as_dict(self):
        """Returns the confusion matrix and labels as an OrderedDict
        """
        retval = OrderedDict()
        retval["labels"] = self.labels
        retval["confusion_matrix"] = ConfusionMatrix.format_matrix(
            self.confusion_matrix)
        return retval

    @classmethod
    def from_dict(cls, json_obj):
        assert "labels" in json_obj, \
            "Couldn't find labels in json_obj"
        assert "confusion_matrix" in json_obj, \
            "Couldn't find confusion_matrix in json_obj"
        retval = cls()
        setattr(retval, "labels", json_obj["labels"])
        setattr(retval, "confusion_matrix", json_obj["confusion_matrix"])
        return retval

    @classmethod
    def format(cls, num, fmt="{0:3.1f}"):
        """Format numbers nicely
        """
        return float(fmt.format(num))

    @classmethod
    def format_matrix(cls, matrix):
        """Pretty print the confusion matrix
        """
        matrix_list = matrix.tolist()
        pretty_list = [[ConfusionMatrix.format(x) for x in row]
                       for row in matrix_list]
        return pretty_list

    @classmethod
    def from_preds(cls, data, label_indexer):
        """Instantiates the confusion matrix
        from the data, and the label_indexer
        """
        assert "preds" in data[0],\
            "Expected preds in data"
        assert "golds" in data[0],\
            "Expected golds in data"
        retval = cls()
        retval.load_data(data, label_indexer)
        return retval
