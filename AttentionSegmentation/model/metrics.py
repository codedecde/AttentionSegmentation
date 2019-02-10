from collections import OrderedDict
from overrides import overrides
from torch import LongTensor
import pdb

from allennlp.training.metrics import \
    BooleanAccuracy

from allennlp.training.metrics.metric import Metric


class ClassificationMetrics(Metric):
    def __init__(self, label_indexer):
        self.num_labels = label_indexer.get_num_tags()
        self.label_indexer = label_indexer
        self.accuracies = OrderedDict()
        for ix in range(self.num_labels - 1):
            tag = self.label_indexer.get_tag(ix)
            classifier = BooleanAccuracy()
            self.accuracies[tag] = classifier

    def __call__(self, pred_labels, gold_labels):
        """
        Arguments:
            pred_labels (Tensor: batch x num_labels - 1):
                The predicted labels (excluding the O tag)
            gold_labels (Tensor: batch x num_labels - 1):
                The gold labels
        Returns:
            None

        Updates the internal state of all the classifiers
        """
        assert (self.num_labels - 1 == pred_labels.size(1))
        for ix, tag in enumerate(self.accuracies):
            # FIXME: Remove this assertion
            assert tag == self.label_indexer.get_tag(ix)
            self.accuracies[tag](pred_labels[:, ix], gold_labels[:, ix])

    def get_metric(self, reset: bool = False):
        """
        Returns
            summary (`OrderedDict`) : The global accuracies
        """
        summary = OrderedDict()
        acc = 0.
        for tag in self.accuracies:
            label_accuracy = self.accuracies[tag].get_metric(reset)
            acc += label_accuracy
            summary[tag] = label_accuracy
        acc /= (self.num_labels - 1)
        summary["accuracy"] = acc
        if reset:
            self.reset()
        return summary

    @overrides
    def reset(self):
        for tag in self.accuracies:
            self.accuracies[tag].reset()
