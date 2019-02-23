from __future__ import absolute_import
import logging
import os
import shutil
import json
from collections import deque
import time
import re
import datetime
import traceback
import numpy as np
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any, Set
import pdb

import torch
import torch.optim.lr_scheduler
from torch.nn.parallel import replicate, parallel_apply
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from tensorboardX import SummaryWriter


from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import peak_memory_mb, gpu_memory_mb
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer

from AttentionSegmentation.commons.trainer_utils import is_sparse,\
    sparse_clip_norm, move_optimizer_to_cuda, TensorboardWriter
# from AttentionSegmentation.visualization.visualize_attns \
#     import html_visualizer
from AttentionSegmentation.model.attn2labels import BasePredictionClass
from AttentionSegmentation.commons.custom_iterators import CustomIterator
logger = logging.getLogger(__name__)

TQDM_COLUMNS = 200


class Trainer(object):
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 iterator: CustomIterator,
                 train_dataset: Dict[str, Iterable[Instance]],
                 validation_dataset: Optional[Dict[str, Iterable[Instance]]] = None,
                 segmenter: Optional[Dict[BasePredictionClass]] = None,
                 patience: Optional[int] = None,
                 validation_metric: str = "-loss",
                 num_epochs: int = 20,
                 base_dir: Optional[str] = None,
                 num_serialized_models_to_keep: Optional[int] = None,
                 cuda_device: Union[int, List] = -1,
                 grad_norm: Optional[float] = None,
                 grad_clipping: Optional[float] = None,
                 learning_rate_scheduler: Optional[LearningRateScheduler] = None
                 ) -> None:
        """
        Parameters
        ----------
        model : ``Model``, required.
            An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
            their ``forward`` method returns a dictionary with a "loss" key, containing a
            scalar tensor representing the loss function to be optimized.
        optimizer : ``torch.nn.Optimizer``, required.
            An instance of a Pytorch Optimizer, instantiated with the parameters of the
            model to be optimized.
        iterator : ``DataIterator``, required.
            A method for iterating over a ``Dataset``, yielding padded indexed batches.
        train_dataset : ``Dataset``, required.
            A ``Dataset`` to train on. The dataset should have already been indexed.
        validation_dataset : ``Dataset``, optional, (default = None).
            A ``Dataset`` to evaluate on. The dataset should have already been indexed.
        segmenter : ``BasePredictionClass`` for converting attention segmentation to labels.
            None if we don't want to visualize
        patience : Optional[int] > 0, optional (default=None)
            Number of epochs to be patient before early stopping: the training is stopped
            after ``patience`` epochs with no improvement. If given, it must be ``> 0``.
            If None, early stopping is disabled.
        validation_metric : str, optional (default="loss")
            Validation metric to measure for whether to stop training using patience
            and whether to serialize an ``is_best`` model each epoch. The metric name
            must be prepended with either "+" or "-", which specifies whether the metric
            is an increasing or decreasing function.
            Allowed values are -loss, +precision-overall, +recall-overall, +f1-measure-overall
        num_epochs : int, optional (default = 20)
            Number of training epochs.
        base_dir : str, optional (default=None)
            Path to directory for the experiment. Models will be saved at 
            {base_dir}/models, and won't be saved if this parameter is not passed.
        num_serialized_models_to_keep: int, optional (default=None)
            The number of models to keep during training
        cuda_device : ``int``, optional (default = -1)
            An integer specifying the CUDA device to use. If -1, the CPU is used.
        grad_norm : ``float``, optional, (default = None).
            If provided, gradient norms will be rescaled to have a maximum of this value.
        grad_clipping : ``float``, optional (default = ``None``).
            If provided, gradients will be clipped `during the backward pass` to have an (absolute)
            maximum of this value.  If you are getting ``NaNs`` in your gradients during training
            that are not solved by using ``grad_norm``, you may need this.
        learning_rate_scheduler : ``PytorchLRScheduler``, optional, (default = None)
            A Pytorch learning rate scheduler. The learning rate will be decayed with respect to
            this schedule at the end of each epoch. If you use
            :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`, this will use the ``validation_metric``
            provided to determine if learning has plateaued.  To support updating the learning
            rate on every batch, this can optionally implement ``step_batch(batch_num_total)`` which
            updates the learning rate given the batch number.
        visualization_dirname : The directory to save the visualization in

        """
        self._model = model
        self._iterator = iterator
        self._optimizer = optimizer
        self._train_data = train_dataset
        self._validation_data = validation_dataset
        self._segmenter = segmenter
        self._base_dir = base_dir

        if patience is None:  # no early stopping
            if validation_dataset:
                logger.warning('You provided a validation dataset but patience was set to None, '
                               'meaning that early stopping is disabled')
        elif (not isinstance(patience, int)) or patience <= 0:
            raise ConfigurationError('{} is an invalid value for "patience": it must be a positive integer '
                                     'or None (if you want to disable early stopping)'.format(patience))
        self._patience = patience
        self._num_epochs = num_epochs

        serialization_dir = None
        if base_dir is not None:
            serialization_dir = os.path.join(base_dir, "models")
        self._serialization_dir = serialization_dir
        assert os.path.exists(self._serialization_dir), "Directory {0} does not exist".format(self._serialization_dir)

        self._serialized_paths = deque()
        # self._last_permanent_saved_checkpoint_time = time.time()
        # self._model_save_interval = model_save_interval

        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping
        self._learning_rate_scheduler = learning_rate_scheduler

        increase_or_decrease = validation_metric[0]
        if increase_or_decrease not in ["+", "-"]:
            raise ConfigurationError("Validation metrics must specify whether they should increase "
                                     "or decrease by pre-pending the metric name with a +/-.")
        self._validation_metric = validation_metric[1:]
        self._validation_metric_decreases = increase_or_decrease == "-"

        if not isinstance(cuda_device, int) and not isinstance(cuda_device, list):
            raise ConfigurationError("Expected an int or list for cuda_device, got {}".format(cuda_device))

        if isinstance(cuda_device, list):
            logger.info(f"WARNING: Multiple GPU support is experimental not recommended for use. "
                        "In some cases it may lead to incorrect results or undefined behavior.")
            self._multiple_gpu = True
            self._cuda_devices = cuda_device
            # data_parallel will take care of transfering to cuda devices,
            # so the iterator keeps data on CPU.
            self._iterator_device = -1
        else:
            self._multiple_gpu = False
            self._cuda_devices = [cuda_device]
            self._iterator_device = cuda_device

        if self._cuda_devices[0] != -1:
            self._model = self._model.cuda(self._cuda_devices[0])

        self._log_interval = 10  # seconds

        self._batch_num_total = 0

        self._last_log = 0.0  # time of last logging
        self._num_serialized_models_to_keep = num_serialized_models_to_keep

    def _enable_gradient_clipping(self) -> None:
        if self._grad_clipping is not None:
            # Pylint is unable to tell that we're in the case that _grad_clipping is not None...
            # pylint: disable=invalid-unary-operand-type
            clip_function = lambda grad: grad.clamp(-self._grad_clipping, self._grad_clipping)
            for parameter in self._model.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(clip_function)

    def _rescale_gradients(self) -> Optional[float]:
        """
        Performs gradient rescaling.
        Is a no-op if gradient rescaling is not enabled.
        """
        if self._grad_norm:
            parameters_to_clip = [p for p in self._model.parameters()
                                  if p.grad is not None]
            return sparse_clip_norm(parameters_to_clip, self._grad_norm)
        return None

    def _data_parallel(self, batch):
        """
        Do the forward pass using multiple GPUs.  This is a simplification
        of torch.nn.parallel.data_parallel to support the allennlp model
        interface.
        """
        inputs, module_kwargs = scatter_kwargs((), batch,
                                               self._cuda_devices, 0)
        used_device_ids = self._cuda_devices[:len(inputs)]
        replicas = replicate(self._model, used_device_ids)
        outputs = parallel_apply(replicas, inputs,
                                 module_kwargs, used_device_ids)

        # Only the 'loss' is needed.
        # a (num_gpu, ) tensor with loss on each GPU
        losses = gather([output['loss']
                         for output in outputs], used_device_ids[0], 0)
        return {'loss': losses.mean()}

    def _batch_loss(self, batch: torch.Tensor,
                    for_training: bool) -> torch.Tensor:
        """
        Does a forward pass on the given batch
        and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.
        """
        if self._multiple_gpu:
            output_dict = self._data_parallel(batch)
        else:
            output_dict = self._model(**batch)

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

    def _get_metrics(self, total_loss: float,
                     num_batches: int, reset: bool = False) -> Dict[str, float]:
        """
        Gets the metrics but sets ``"loss"`` to
        the total loss divided by the ``num_batches`` so that
        the ``"loss"`` metric is "average loss per batch".
        """
        metrics = self._model.get_metrics(reset=reset)
        metrics["loss"] = float(total_loss / num_batches) if num_batches > 0 else 0.0
        return metrics

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info(f"Peak CPU memory usage MB: {peak_memory_mb()}")
        if torch.cuda.is_available():
            for gpu, memory in gpu_memory_mb().items():
                logger.info(f"GPU {gpu} memory usage MB: {memory}")

        train_loss = 0.0
        # Set the model to "train" mode.
        self._model.train()

        # Get tqdm for the training batches
        train_generator = self._iterator(self._train_data,
                                         num_epochs=1,
                                         cuda_device=self._iterator_device)
        num_training_batches = self._iterator.get_num_batches(self._train_data)
        train_generator_tqdm = Tqdm.tqdm(train_generator,
                                         total=num_training_batches
                                         )
        self._last_log = time.time()
        last_save_time = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        for batch in train_generator_tqdm:
            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            self._optimizer.zero_grad()
            loss = self._batch_loss(batch, for_training=True)
            loss.backward()

            # Make sure Variable is on the cpu before converting to numpy.
            # .cpu() is a no-op if you aren't using GPUs.
            train_loss += loss.data.cpu().numpy()

            batch_grad_norm = self._rescale_gradients()

            # This does nothing if batch_num_total is None or you are using an
            # LRScheduler which doesn't update per batch.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(batch_num_total)
            self._optimizer.step()

            # Update the description with the latest metrics
            metrics = self._get_metrics(train_loss, batches_this_epoch)
            description = self._description_from_metrics(metrics)

            train_generator_tqdm.set_description(description, refresh=False)
            if hasattr(self, "_tf_params") and self._tf_params is not None:
                # We have TF logging
                if self._batch_num_total % self._tf_params["log_every"] == 0:
                    self._tf_log(metrics, self._batch_num_total)

        return self._get_metrics(train_loss, batches_this_epoch, reset=True)

    def _should_stop_early(self, metric_history: List[float]) -> bool:
        """
        uses patience and the validation metric to determine if training should stop early
        """
        if self._patience and self._patience < len(metric_history):
            # Pylint can't figure out that in this branch `self._patience` is an int.
            # pylint: disable=invalid-unary-operand-type

            # Is the best score in the past N epochs worse than the best score overall?
            if self._validation_metric_decreases:
                return min(metric_history[-self._patience:]) > min(metric_history)
            else:
                return max(metric_history[-self._patience:]) < max(metric_history)

        return False

    def _metrics_to_console(self,  # pylint: disable=no-self-use
                            train_metrics: dict,
                            val_metrics: dict = None) -> None:
        """
        Logs all of the train metrics (and validation metrics, if provided) to the console.
        """
        val_metrics = val_metrics or {}
        dual_message_template = "Training %s : %3f    Validation %s : %3f "
        message_template = "%s %s : %3f "

        metric_names = set(sorted(train_metrics.keys()))
        if val_metrics:
            metric_names.update(val_metrics.keys())

        write_buf = "Metrics:\n"

        for name in metric_names:
            train_metric = train_metrics.get(name)
            val_metric = val_metrics.get(name)
            if val_metric is not None and train_metric is not None:
                training = "Training"
                validation = "Validation"
                write_buf += f"{training:>32s} {name:18s}: {train_metric:4.3f}  "
                write_buf += f"{validation} {name:18s}: {val_metric:4.3f}\n"
            elif val_metric is not None:
                # logger.info(message_template, "Validation", name, val_metric)
                write_buf += "{0:>32s} {1:18s}: {2:4.3f}".format(
                    "Validation", name, val_metric)
            elif train_metric is not None:
                # logger.info(message_template, "Training", name, train_metric)
                write_buf += "{0:>32s} {1:18s}: {2:4.3f}".format(
                    "Training", name, train_metric)
        logger.info(write_buf)

    def test(self, data):
        """
            Computes the metrics for data
        """
        model_path = os.path.join(self._serialization_dir, "best.th")
        logger.info("Loading best model from {0}".format(model_path))
        model_state = torch.load(model_path,
                                 map_location=util.device_mapping(-1))
        self._model.load_state_dict(model_state)
        loss, num_batches = self._inference_loss(data, logger_string="Testing")
        metrics = self._get_metrics(loss, num_batches, reset=True)
        metric_names = metrics.keys()
        message_template = "%s %s : %3f"
        for name in metric_names:
            test_metric = metrics.get(name)
            logger.info(message_template, "Test", name, test_metric)
        return metrics


    def _inference_loss(self, data,
                        logger_string="validation") -> Tuple[float, int]:
        """
        Computes the loss on the data passed.
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
        logger.info("{0} done".format(logger_string))

        return inference_loss, batches_this_epoch

    def _get_validation_metric(self, val_metrics):
        return val_metrics[self._validation_metric]

    def train(self, continue_training=False) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.

        Parameters
        ------------
        continue_training: bool (default=False)
            Load a checkpoint and start training
        """
        epoch_counter, validation_metric_per_epoch = 0, []
        if continue_training:
            try:
                epoch_counter, validation_metric_per_epoch = self._restore_checkpoint()
            except RuntimeError:
                traceback.print_exc()
                raise ConfigurationError("Could not recover training from the checkpoint.  Did you mean to output to "
                                         "a different serialization directory or delete the existing serialization "
                                         "directory?")

        self._enable_gradient_clipping()

        logger.info("Beginning training.")

        train_metrics: Dict[str, float] = {}
        val_metrics: Dict[str, float] = {}
        epochs_trained = 0
        training_start_time = time.time()
        for epoch in range(epoch_counter, self._num_epochs):
            logger.info("=" * 50)
            logger.info("Starting Training Epoch %d/%d",
                        epoch + 1, self._num_epochs)
            epoch_start_time = time.time()
            train_metrics = self._train_epoch(epoch)

            if self._validation_data is not None:
                # We have a validation set, so compute all the metrics on it.
                val_loss, num_batches = self._inference_loss(
                    self._validation_data, logger_string="Validation")
                val_metrics = self._get_metrics(
                    val_loss, num_batches, reset=True)

                # Check validation metric for early stopping
                this_epoch_val_metric = self._get_validation_metric(
                    val_metrics)
                validation_metric_per_epoch.append(this_epoch_val_metric)
                if self._should_stop_early(validation_metric_per_epoch):
                    logger.info("Ran out of patience. Stopping training.")
                    break

                # Check validation metric to see if it's the best so far
                if self._validation_metric_decreases:
                    is_best_so_far = this_epoch_val_metric == min(
                        validation_metric_per_epoch)
                else:
                    is_best_so_far = this_epoch_val_metric == max(
                        validation_metric_per_epoch)
            else:
                # No validation set, so just assume it's the best so far.
                is_best_so_far = True
                val_metrics = {}
                this_epoch_val_metric = None

            self._save_checkpoint(
                epoch, validation_metric_per_epoch, is_best=is_best_so_far)
            self._metrics_to_console(train_metrics, val_metrics)
            # Now the predictions and visualization
            if self._segmenter is not None and is_best_so_far:
                visualization_file = os.path.join(
                    self._base_dir, "visualization", "validation.html")
                prediction_file = os.path.join(
                    self._base_dir, "predictions.json")
                logger.info(
                    f"Writing validation visualization at {visualization_file}"
                )
                self._segmenter.get_predictions(
                    instances=self._validation_data,
                    model=self._model,
                    cuda_device=self._iterator_device,
                    prediction_file=prediction_file,
                    visualization_file=visualization_file,
                    verbose=True)
                logger.info(f"Writing predictions to {prediction_file}")

            if self._learning_rate_scheduler:
                # The LRScheduler API is agnostic to whether your schedule requires a validation metric -
                # if it doesn't, the validation metric passed here is ignored.
                old_lr = min(param_group['lr']
                             for param_group in self._optimizer.param_groups)
                self._learning_rate_scheduler.step(
                    this_epoch_val_metric, epoch)
                new_lr = min(param_group['lr']
                             for param_group in self._optimizer.param_groups)
                logger.info("Reducing LR: {0:.2e} -> {1:.2e}".format(
                    old_lr, new_lr))

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s",
                        time.strftime("%H:%M:%S",
                                      time.gmtime(epoch_elapsed_time)))

            if epoch < self._num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * \
                    ((self._num_epochs - epoch_counter) / float(
                        epoch - epoch_counter + 1) - 1)
                formatted_time = time.strftime(
                    "%H:%M:%S", time.gmtime(estimated_time_remaining))
                logger.info("Estimated training time remaining: %s",
                            formatted_time)

            epochs_trained += 1
            logger.info("=" * 50)

        training_elapsed_time = time.time() - training_start_time
        metrics = {
            "training_duration": time.strftime(
                "%H:%M:%S", time.gmtime(training_elapsed_time)),
            "training_start_epoch": epoch_counter,
            "training_epochs": epochs_trained
        }
        for key, value in train_metrics.items():
            metrics["training_" + key] = value
        for key, value in val_metrics.items():
            metrics["validation_" + key] = value

        if validation_metric_per_epoch:
            # We may not have had validation data, so we need to hide this behind an if.
            if self._validation_metric_decreases:
                best_validation_metric = min(validation_metric_per_epoch)
            else:
                best_validation_metric = max(validation_metric_per_epoch)
            best_val_key = f"best_validation_{self._validation_metric}"
            metrics[best_val_key] = best_validation_metric
            metrics['best_epoch'] = [
                i for i, value in enumerate(validation_metric_per_epoch)
                if value == best_validation_metric][-1]
        return metrics

    def _description_from_metrics(self, metrics: Dict[str, float]) -> str:
        # pylint: disable=no-self-use
        return ', '.join(["%s: %.4f" % (name, value)
                          for name, value in metrics.items()]) + " ||"

    def _save_checkpoint(self,
                         epoch: Union[int, str],
                         val_metric_per_epoch: List[float],
                         is_best: Optional[bool] = None) -> None:
        """
        Saves a checkpoint of the model to self._serialization_dir.
        Is a no-op if self._serialization_dir is None.

        Parameters
        ----------
        epoch : Union[int, str], required.
            The epoch of training.  If the checkpoint is saved in the middle
            of an epoch, the parameter is a string with the epoch and timestamp.
        is_best: bool, optional (default = None)
            A flag which causes the model weights at the given epoch to
            be copied to a "best.th" file. The value of this flag should
            be based on some validation metric computed by your model.
        """
        if self._serialization_dir is not None:
            model_path = os.path.join(self._serialization_dir,
                                      "model_state_epoch_{}.th".format(epoch))
            model_state = self._model.state_dict()
            torch.save(model_state, model_path)

            training_state = {'epoch': epoch,
                              'val_metric_per_epoch': val_metric_per_epoch,
                              'optimizer': self._optimizer.state_dict(),
                              'batch_num_total': self._batch_num_total}
            training_path = os.path.join(
                self._serialization_dir,
                "training_state_epoch_{}.th".format(epoch))
            torch.save(training_state, training_path)
            if is_best:
                logger.info("Best validation performance so far. "
                            "Copying weights to '%s/best.th'.", self._serialization_dir)
                shutil.copyfile(
                    model_path, os.path.join(
                        self._serialization_dir, "best.th"))

            if self._num_serialized_models_to_keep:
                self._serialized_paths.append(
                    [time.time(), model_path, training_path])
                if len(self._serialized_paths) > self._num_serialized_models_to_keep:
                    paths_to_remove = self._serialized_paths.popleft()
                    for fname in paths_to_remove[1:]:
                        os.remove(fname)

    def _restore_checkpoint(self) -> Tuple[int, List[float]]:
        """
        Restores a model from a serialization_dir to the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from  model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

        If ``self._serialization_dir`` does not exist or does not contain any checkpointed weights,
        this function will do nothing and return 0.

        Returns
        -------
        epoch: int
            The epoch at which to resume training, which should be one after the epoch
            in the saved training state.
        """
        have_checkpoint = (
            self._serialization_dir is not None and
            any(
                "model_state_epoch_" in x
                for x in os.listdir(os.path.join(self._serialization_dir))
            )
        )
        if not have_checkpoint:
            # No checkpoint to restore, start at 0
            logger.info("Didn't find any models to restore."
                        "Starting training from scratch")
            return 0, []

        serialization_files = os.listdir(
            os.path.join(self._serialization_dir))
        model_checkpoints = [x for x in serialization_files
                             if "model_state_epoch" in x]
        # Get the last checkpoint file.  Epochs are specified as either an
        # int (for end of epoch files) or with epoch and timestamp for
        # within epoch checkpoints, e.g. 5.2018-02-02-15-33-42
        found_epochs = [
            # pylint: disable=anomalous-backslash-in-string
            re.search("model_state_epoch_([0-9\.\-]+)\.th", x).group(1)
            for x in model_checkpoints
        ]
        int_epochs: Any = []
        for epoch in found_epochs:
            pieces = epoch.split('.')
            if len(pieces) == 1:
                # Just a single epoch without timestamp
                int_epochs.append([int(pieces[0]), 0])
            else:
                # has a timestamp
                int_epochs.append([int(pieces[0]), pieces[1]])
        last_epoch = sorted(int_epochs, reverse=True)[0]
        if last_epoch[1] == 0:
            epoch_to_load = str(last_epoch[0])
        else:
            epoch_to_load = '{0}.{1}'.format(last_epoch[0], last_epoch[1])

        model_path = os.path.join(
            self._serialization_dir,
            "model_state_epoch_{}.th".format(epoch_to_load))
        training_state_path = os.path.join(
            self._serialization_dir,
            "training_state_epoch_{}.th".format(epoch_to_load))

        # Load the parameters onto CPU, then transfer to GPU.
        # This avoids potential OOM on GPU for large models that
        # load parameters onto GPU then make a new GPU copy into the parameter
        # buffer. The GPU transfer happens implicitly in load_state_dict.
        model_state = torch.load(
            model_path, map_location=util.device_mapping(-1))
        training_state = torch.load(
            training_state_path, map_location=util.device_mapping(-1))
        self._model.load_state_dict(model_state)
        self._optimizer.load_state_dict(training_state["optimizer"])
        move_optimizer_to_cuda(self._optimizer)

        # We didn't used to save `validation_metric_per_epoch`, so we can't assume
        # that it's part of the trainer state. If it's not there, an empty list is all
        # we can do.
        if "val_metric_per_epoch" not in training_state:
            logger.warning("trainer state `val_metric_per_epoch`"
                           "not found, using empty list")
            val_metric_per_epoch: List[float] = []
        else:
            val_metric_per_epoch = training_state["val_metric_per_epoch"]

        if isinstance(training_state["epoch"], int):
            epoch_to_return = training_state["epoch"] + 1
        else:
            epoch_to_return = int(training_state["epoch"].split('.')[0]) + 1

        # For older checkpoints with batch_num_total missing, default to old behavior where
        # it is unchanged.
        batch_num_total = training_state.get('batch_num_total')
        if batch_num_total is not None:
            self._batch_num_total = batch_num_total

        return epoch_to_return, val_metric_per_epoch

    @classmethod
    def get_args(cls,
                 model: Model,
                 base_dir: str,
                 iterator: DataIterator,
                 train_data: Iterable[Instance],
                 validation_data: Optional[Iterable[Instance]],
                 segmenter: Optional[BasePredictionClass],
                 params: Params) -> Dict[str, Any]:
        patience = params.pop_int("patience", None)
        validation_metric = params.pop("validation_metric", "-loss")
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = params.pop_int("cuda_device", -1)
        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)
        lr_scheduler_params = params.pop("learning_rate_scheduler", None)
        num_serialized_models_to_keep = params.pop(
            "num_serialized_models_to_keep", None)

        if cuda_device >= 0:
            model = model.cuda(cuda_device)
        parameters = [[n, p] for n, p in model.named_parameters()
                      if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))

        if lr_scheduler_params:
            scheduler = LearningRateScheduler.from_params(
                optimizer, lr_scheduler_params)
        else:
            scheduler = None
        params.assert_empty(cls.__name__)
        kwargs = {}
        kwargs['model'] = model
        kwargs['optimizer'] = optimizer
        kwargs['iterator'] = iterator
        kwargs['train_dataset'] = train_data
        kwargs['validation_dataset'] = validation_data
        kwargs['segmenter'] = segmenter
        kwargs['patience'] = patience
        kwargs['validation_metric'] = validation_metric
        kwargs['num_epochs'] = num_epochs
        kwargs['base_dir'] = base_dir
        kwargs['cuda_device'] = cuda_device
        kwargs['grad_norm'] = grad_norm
        kwargs['grad_clipping'] = grad_clipping
        kwargs['learning_rate_scheduler'] = scheduler
        kwargs['num_serialized_models_to_keep'] = num_serialized_models_to_keep
        return kwargs

    @classmethod
    def from_params(cls, *args, **kwargs) -> 'Trainer':
        new_args = cls.get_args(*args, **kwargs)
        return Trainer(**new_args)

