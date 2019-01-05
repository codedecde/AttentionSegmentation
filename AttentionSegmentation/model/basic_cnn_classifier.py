from __future__ import absolute_import
import torch
import numpy as np
from torch.nn import Linear
from torch.nn import Dropout
import torch.nn.functional as F
from overrides import overrides
from typing import Dict, Optional
from collections import OrderedDict

import allennlp.nn.util as util
from allennlp.common.params import Params
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder
from allennlp.data import Vocabulary
from allennlp.common.checks import check_dimensions_match

import model.text_field_embedder as text_field_embedder_module
from model.metrics import ClassificationMetrics

from commons.utils import to_numpy

# FIXME Implement this
from preprocess.label_indices\
    import LabelIndicesBiMap, CondensedLabelIndicesBiMap


class BasicCNNClassifier(Model):
    """This model uses a basic CNN model, followed by maxpooling
    and classification

    For Labels like Cancellation, local structure (bi-grams / tri-grams)
    are very informative, and hence, it makes sense to use a CNN for them.
    We usually use a filter size of (2, 3) to account for
    bi-grams / tri-grams.

    Arguments:
        vocab (``Vocabulary``): The vocabulary used for indexing
        text_field_embedder (``TextFieldEmbedder``):
            The text field embedders
        encoder (``Seq2SeqEncoder``): Encoder for encoding word
            level information.
        thresh (float): The class threshold used for predictions.
            (i.e, probs > thresh are classified as 1)
        dropout (float): The amout of dropout being applied
        label_namespace (str): The namespace of the labels
        label_indexer (str): The Label indexer being used
            (LabelIndicesBiMap|CondensedLabelIndicesBiMap)
        initializer (``InitializerApplicator``): The Initializer
            being used
        regularizer (Optional[``RegularizerApplicator``]): The
            regularizers, if any.

    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        encoder: Seq2VecEncoder,
        thresh: float = 0.5,
        label_namespace: str = "labels",
        dropout: Optional[float] = None,
        label_indexer: str = "LabelIndicesBiMap",
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None
    ) -> None:

        super(BasicCNNClassifier, self).__init__(vocab, regularizer)
        self.label_namespace = label_namespace
        self.label_indexer = getattr(
            text_field_embedder_module, label_indexer)
        # FIXME: Implementh this
        self.num_labels = self.label_indexer.get_num_labels()
        # Prediction thresholds
        self.thresh = thresh
        self.log_thresh = np.log(thresh + 1e-5)
        # Model
        # Text field embedder
        self.text_field_embedder = text_field_embedder
        # CNN Encoder
        self.encoder = encoder
        # Dropout
        if dropout is not None and dropout > 0:
            self.dropout = Dropout(dropout)
        else:
            self.dropout = None
        # Label space projections
        output_dim = self.encoder.get_output_dim()
        self.logits_layer = Linear(output_dim, self.num_labels)
        self.classification_metric = ClassificationMetrics(
            self.num_labels, label_indexer)
        initializer(self)

        # Some dimension checks
        check_dimensions_match(
            text_field_embedder.get_output_dim(), encoder.get_input_dim(),
            "text field embedding dim", "word encoder input dim")

    def apply_dropout(self, tensor):
        """Apply dropout
        """
        if self.dropout:
            tensor = self.dropout(tensor)
        return tensor

    @overrides
    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """Computes the forward pass.

        .. note::

            We use label smoothing of 0.1

        Commonly refered symbols:

            * T : max_seq_len
            * C : max_char_len
            * L : num_labels
            * out_dim : The output dim from the attention layer
            * batch : batch size

        Arguments:
          tokens (Dict[str, ``LongTensor``]): The tokens. Contains

              * tokens: batch x T
              * chars: batch x T x C
              * elmo [Optional]: batch x T x C

          labels (``LongTensor``): batch x L:
              The output labels

        Returns:
            (Dict[str, ``Tensor``]): The outputs. Contains the following:

                * loss: 1 x 1 : The BCE Loss
                * preds: batch x L : The probabilites predicted
                  Note that each is an independant prediction for the class
                  i.e The columns do **not** sum to 1

        """
        outputs = {}
        embedded_text_input = self.text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)  # batch x T
        embedded_text_input = self.apply_dropout(embedded_text_input)
        encoded_text = self.encoder(embedded_text_input, mask)
        encoded_text = self.apply_dropout(encoded_text)
        logits = self.logits_layer(encoded_text).squeeze(-1)
        log_probs = F.logsigmoid(logits)
        outputs["logits"] = logits
        outputs["mask"] = mask
        outputs["log_probs"] = log_probs
        if labels is not None:
            smoothing = 0.1
            soft_labels = labels + smoothing - (2 * smoothing * labels)
            loss = -(soft_labels * log_probs +
                     ((1 - soft_labels) * F.logsigmoid(-logits)))
            loss = loss.mean()
            outputs["loss"] = loss
            pred_labels = log_probs.gt(self.log_thresh).long()
            self.classification_metric(pred_labels, labels)
        return outputs

    @overrides
    def decode(
        self,
        output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        log_probs = output_dict["log_probs"]
        pred_labels = log_probs.gt(self.log_thresh).long()
        pred_labels = to_numpy(pred_labels, True)
        batch, num_labels = pred_labels.shape
        output_dict["labels"] = [
            [self.label_indexer.lookup(label_ix)
             for label_ix in range(num_labels)
             if pred_labels[batch_ix][label_ix] == 1]
            for batch_ix in range(batch)
        ]
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = self.classification_metric.get_metric(reset=reset)
        return OrderedDict({x: y for x, y in metric_dict.items()})

    @classmethod
    def from_params(cls,
                    vocab: Vocabulary,
                    params: Params) -> 'AttentionClassifier':
        """Constructs class from ``Params`` and ``Vocabulary``
        """
        embedder_params = params.pop("text_field_embedder")
        embedder_type = embedder_params.get("type", None)
        if embedder_type is None:
            text_field_embedder = TextFieldEmbedder.from_params(
                vocab, embedder_params
            )
        else:
            embedder_type = embedder_params.pop("type")
            text_field_embedder = getattr(
                text_field_embedder_module, embedder_type).from_params(
                vocab, embedder_params
            )

        encoder = Seq2VecEncoder.from_params(params.pop("encoder"))

        label_namespace = params.pop("label_namespace", "labels")

        dropout = params.pop("dropout", None)

        threshold = params.pop("threshold", 0.5)
        initializer = InitializerApplicator.from_params(
            params.pop('initializer', [])
        )
        regularizer = RegularizerApplicator.from_params(
            params.pop('regularizer', [])
        )
        label_indexer = params.pop("label_indexer", "LabelIndicesBiMap")

        params.assert_empty(cls.__name__)

        return cls(
            vocab=vocab,
            text_field_embedder=text_field_embedder,
            encoder=encoder,
            thresh=threshold,
            label_namespace=label_namespace,
            dropout=dropout,
            label_indexer=label_indexer,
            initializer=initializer,
            regularizer=regularizer
        )
