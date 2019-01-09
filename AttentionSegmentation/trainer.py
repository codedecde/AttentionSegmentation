import os
import torch
from collections import defaultdict
from typing import Dict, Tuple
from overrides import overrides
import numpy as np
import logging
import re

from AttentionSegmentation.allennlp.nn import util
from AttentionSegmentation.allennlp.common.tqdm import Tqdm
import AttentionSegmentation.commons.trainer as common_trainer

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
    def __init__(self, *args, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)
        self._reset_counter()

    def _reset_counter(self):
        self._zero_counts = {"zero": 0., "num_preds": 0}

    @overrides
    def train(self, *args, **kwargs):
        super(Trainer, self).train(*args, **kwargs)

    @overrides
    def _description_from_metrics(
            self, metrics: Dict[str, Dict[str, float]]) -> str:
        """Show metrics.

        The metrics format we get are of the form::

            "Label"
                "precision": _
                "recall": _
                "fscore": _
                "accuracy": _
                "loss": _

        """
        new_metrics = defaultdict(list)
        for label in metrics:
            if re.match(".*loss$", label) is not None:
                # all losses are of the form loss/ entropy_loss, etc
                new_metrics[label] = metrics[label]
            else:
                for metric in metrics[label]:
                    new_metrics[metric].append(metrics[label][metric])
        metrics_to_show = ["accuracy", "fscore"]
        losses_to_show = ["entropy_loss", "coverage_loss"]
        writebuf = []
        for metric in metrics_to_show:
            writebuf.append(f"{metric}: {np.mean(new_metrics[metric]):2.2f}")
        for loss in losses_to_show:
            if loss in new_metrics:
                writebuf.append(
                    get_loss_string(loss, new_metrics[loss]))
        lossval = new_metrics['loss']
        writebuf.append(get_loss_string("loss", lossval))
        write_string = ", ".join(writebuf) + " ||"
        return write_string

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
        loss, num_batches = self._inference_loss(data, logger_string="Testing")
        metrics = self._get_metrics(loss, num_batches, reset=True)
        writebuf = "########## TESTING ##########"
        writebuf = self._prettyprint(metrics, writebuf)
        logger.info(writebuf)

    @overrides
    def _get_validation_metric(self, val_metrics):
        """We return the mean of all _validation_metric
        across labels.
        """
        # returns average metric across all labels
        if self._validation_metric in val_metrics.keys():
            return val_metrics[self._validation_metric]
        ret_metric_list = []
        for label in val_metrics:
            if re.match(".*loss$", label) is not None:
                continue
            ret_metric_list.append(val_metrics[label][self._validation_metric])
        return np.mean(ret_metric_list)

    def _prettyprint(self, metrics, writebuf):
        """Prettily printing the precision, recall, fscore for each label.
        """
        writebuf += "\n{0:>20s} {1:>10s} {2:>10s} {3:>10s} {4:>10s} {5:>10s}".format(
            "Tag", "Precision", "Recall", "Fscore", "Accuracy", "Loss")
        averages = {"precision": [],
                    "recall": [], "fscore": [], "accuracy": []}
        for label in sorted(metrics.keys()):
            if re.match(".*loss$", label) is not None:
                continue
            precision = "{0:.2f}".format(metrics[label]["precision"])
            averages["precision"].append(metrics[label]["precision"])
            recall = "{0:.2f}".format(metrics[label]["recall"])
            averages["recall"].append(metrics[label]["recall"])
            fscore = "{0:.2f}".format(metrics[label]["fscore"])
            averages["fscore"].append(metrics[label]["fscore"])
            accuracy = "{0:.2f}".format(metrics[label]["accuracy"])
            averages["accuracy"].append(metrics[label]["accuracy"])
            loss = "{0:.2e}".format(metrics["loss"])
            writebuf += "\n{0:>20s} {1:>10s} {2:>10s} {3:>10s} {4:>10s} {5:>10s}".format(
                label, precision, recall,
                fscore, accuracy, loss)
        precision = "{0:.2f}".format(np.mean(averages["precision"]))
        recall = "{0:.2f}".format(np.mean(averages["recall"]))
        fscore = "{0:.2f}".format(np.mean(averages["fscore"]))
        accuracy = "{0:.2f}".format(np.mean(averages["accuracy"]))
        writebuf += "\n{0:>20s} {1:>10s} {2:>10s} {3:>10s} {4:>10s}".format(
            "Mean", precision, recall,
            fscore, accuracy
        )
        writebuf += "\n############################"
        return writebuf

    @overrides
    def _batch_loss(self, batch: torch.Tensor,
                    for_training: bool) -> torch.Tensor:
        """Does a forward pass on the given batch
        and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.
        """
        if self._multiple_gpu:
            output_dict = self._data_parallel(batch)
        else:
            output_dict = self._model(**batch)

        if not for_training:
            thresh = np.log(0.5)
            num_zero_preds = (output_dict['log_probs'].gt(thresh).long() == 0).sum().data.cpu().numpy()[0]
            self._zero_counts["zero"] += num_zero_preds
            self._zero_counts["num_preds"] += output_dict['log_probs'].numel()
        try:
            loss = output_dict["loss"]
            if for_training:
                loss += self._model.get_regularization_penalty()
        except KeyError:
            if for_training:
                raise RuntimeError("The model you are trying to optimize does not contain a"
                                   " 'loss' key in the output of model.forward(inputs).")
            loss = None

        return loss

    @overrides
    def _metrics_to_console(self,  # pylint: disable=no-self-use
                            train_metrics: dict,
                            val_metrics: dict = None) -> None:
        """Log the training / validation metric
        """
        writebuf = "########## TRAINING ##########"
        writebuf = self._prettyprint(train_metrics, writebuf)
        logger.info(writebuf)
        if val_metrics:
            writebuf = "########## VALIDATION ##########"
            writebuf = self._prettyprint(val_metrics, writebuf)
            logger.info(writebuf)

    @classmethod
    @overrides
    def from_params(cls, *args, **kwargs) -> 'Trainer':
        new_args = cls.get_args(*args, **kwargs)
        return Trainer(**new_args)

    @overrides
    def _inference_loss(self, data,
                        logger_string="validation") -> Tuple[float, int]:
        """Computes the loss on the data passed.
        Returns it and the number of batches.
        Is primarily used for validation and testing
        """
        logger.info("Starting with {0}".format(logger_string))

        self._model.eval()

        inference_generator = self._iterator(data,
                                             num_epochs=1,
                                             cuda_device=self._iterator_device,
                                             for_training=False)
        num_batches = self._iterator.get_num_batches(data)
        inference_generator_tqdm = Tqdm.tqdm(inference_generator,
                                             total=num_batches
                                             )
        batches_this_epoch = 0
        inference_loss = 0
        self._reset_counter()
        for batch in inference_generator_tqdm:

            loss = self._batch_loss(batch, for_training=False)
            if loss is not None:
                # You shouldn't necessarily have to compute a loss for validation, so we allow for
                # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                # currently only used as the divisor for the loss function, so we can safely only
                # count those batches for which we actually have a loss.  If this variable ever
                # gets used for something else, we might need to change things around a bit.
                batches_this_epoch += 1
                inference_loss += loss.data.cpu().numpy()

            # Update the description with the latest metrics
            inference_metrics = self._get_metrics(inference_loss, batches_this_epoch)
            description = self._description_from_metrics(inference_metrics)
            inference_generator_tqdm.set_description(description, refresh=False)
        num_zero_preds = self._zero_counts["zero"]
        num_preds = self._zero_counts["num_preds"]
        logger.info("{0} done. ({1} / {2}) zero predicted".format(
            logger_string, num_zero_preds, num_preds))

        return inference_loss, batches_this_epoch
