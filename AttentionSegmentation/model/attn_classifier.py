from __future__ import absolute_import
import torch
import numpy as np
import re
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
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.data import Vocabulary
from allennlp.common.checks import check_dimensions_match
from allennlp.training.metrics import Average

# from Model.attention_module import BaseAttention,\
#     DotAttention, BahdanauAttention
import AttentionSegmentation.model.attention_module as attention_modules
# from AttentionSegmentation.Model.text_field_embedder import GatedTextFieldEmbedder
import AttentionSegmentation.model.text_field_embedder as text_field_embedder_modules
from AttentionSegmentation.model.metrics import ClassificationMetrics

from AttentionSegmentation.commons.utils import to_numpy



class AttentionClassifier(Model):
    """The basic attention classifier.

    This computes num_label attention distributions
    over the entire text (i.e a flat attention distribution).
    Also has a bunch of auxillary loss function to promote
    certain desired behaviors.

    Arguments:
        vocab (``Vocabulary``): The vocabulary used for indexing
        text_field_embedder (``TextFieldEmbedder``):
            The text field embedders
        key_dim (int): The hidden dimensions of keys
        thresh (float): The class threshold used for predictions.
            (i.e, probs > thresh are classified as 1)
        encoder (``Seq2SeqEncoder``): Encoder for encoding word
            level information.
        attention_module (``BaseAttention``): The kind of attention
            method being used.
        entropy_scalar (float): The scalar used to weigh the entropy
            loss
        coverage_scalar (float): The scalar used to weigh the
            coverage loss
        label_namespace (str): The namespace of the labels
        dropout (float): The amout of dropout being applied
        label_indexer (str): The Label indexer being used
            (LabelIndicesBiMap|CondensedLabelIndicesBiMap)
        initializer (``InitializerApplicator``): The Initializer
            being used
        regularizer (Optional[``RegularizerApplicator``]): The
            regularizers, if any.

    """

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 key_dim: int,
                 thresh: 0.5,
                 encoder: Seq2SeqEncoder,
                 attention_module: attention_modules.BaseAttention,
                 entropy_scalar: float = 0.,
                 coverage_scalar: float = 0.,
                 label_namespace: str = "labels",
                 dropout: float = None,
                 label_indexer: str = "LabelIndicesBiMap",
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(AttentionClassifier, self).__init__(vocab, regularizer)

        self.entropy_scalar = entropy_scalar
        self.coverage_scalar = coverage_scalar

        self.label_namespace = label_namespace
        self.text_field_embedder = text_field_embedder
        self.label_indexer = eval(label_indexer)
        # FIXME: Implement this
        self.num_labels = self.label_indexer.get_num_labels()
        self.key_dim = key_dim
        self.encoder = encoder
        self.thresh = thresh
        self.log_thresh = np.log(thresh + 1e-5)
        if dropout:
            self.dropout = Dropout(dropout)
        else:
            self.dropout = None
        self.attention_module = attention_module
        output_dim = self.attention_module.get_output_dim()

        self.logits_layer = Linear(output_dim, 1)

        self.classification_metric = ClassificationMetrics(
            self.num_labels, label_indexer)
        if self.entropy_scalar > 0:
            self.entropy_average = Average()
        if self.coverage_scalar > 0:
            self.coverage_average = Average()
        check_dimensions_match(text_field_embedder.get_output_dim(),
                               encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        initializer(self)

    def apply_dropout(self, tensor):
        """Apply dropout
        """
        if self.dropout:
            tensor = self.dropout(tensor)
        return tensor

    def _compute_entropy(self, attns, mask):
        """The entropy loss

          Computes the entropy of the attn distribution.
          Penalising the entropy allows for more peaked
          attention distributions, which is better for
          segmentation.

          Entropy is computed as::

              \\sum_{i} p_i log(p_i)

          Arguments:
              attns (``Tensor``):
                  batch x L x T : The attention probs
              mask (``LongTensor``):
                  batch x T : The context mask
          Returns:
              ``Tensor`` : The entropy (a scalar)

        """
        attns_expanded = attns.transpose(0, -1).contiguous().view(-1)
        entropy_scalars = attns_expanded * torch.log(attns_expanded + 1e-5)
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, 3)
        mask_expanded = mask_expanded.transpose(0, -1).contiguous().view(-1)
        mask_expanded = mask_expanded.unsqueeze(-1)
        non_zero_indices = mask_expanded.view(-1).nonzero().view(-1)
        entropy = torch.gather(entropy_scalars, 0, non_zero_indices).sum()
        return -entropy

    def _compute_coverage(self, attns, mask):
        """The coverage loss

          Computes coverage::

              \\sum_{i=1}^{T} min(a^{i}_{1} ... a^{i}_{L})

          Where T is sequence length, and L is number of labels.
          This promotes each attention mask to attend over different things
          Taken from (https://arxiv.org/abs/1704.04368)

          Arguments:
              attns (``Tensor``):
                  batch x L x T : The attention probs
              mask (``LongTensor``):
                  batch x T : The context mask
          Returns:
              ``Tensor`` : The coverage (a scalar)

        """
        attn_minvals, _ = torch.min(attns, dim=1, keepdim=False)
        attn_minvals_flattened = attn_minvals.contiguous().view(-1)
        non_zero_indices = mask.view(-1).nonzero().view(-1)
        coverage = torch.gather(
            attn_minvals_flattened, 0, non_zero_indices).sum()
        return coverage

    @overrides
    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """This computes the forward pass.

        .. note::

            We use label smoothing of 0.1

        Commonly refered symbols

            * T : max_seq_len
            * C : max_char_len
            * L : num_labels
            * out_dim : The output dim from the attention layer
            * batch : batch size

        Arguments:
          tokens (Dict[str, ``LongTensor``]): The tokens. Contain

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
                * alpha: batch x L x T : The attention weights for each
                  label class

        """
        outputs = {}
        embedded_text_input = self.text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)  # batch x T
        embedded_text_input = self.apply_dropout(embedded_text_input)
        encoded_text = self.encoder(embedded_text_input, mask)
        encoded_text = self.apply_dropout(encoded_text)
        '''
          encoded_text: # batch x T x embed_dim
        '''
        output_embeds, alpha = self.attention_module(
            context=encoded_text,
            mask=mask.long()
        )
        '''
          output_embeds: batch x L x out_dim
          alpha : batch x L x T
        '''
        logits = self.logits_layer(output_embeds).squeeze(-1)  # batch x L
        log_probs = F.logsigmoid(logits)
        # Entropy computation
        if self.entropy_scalar > 0:
            entropy = self._compute_entropy(alpha, mask.float())
            outputs["entropy_loss"] = entropy * self.entropy_scalar
            entropy_scalar = to_numpy(entropy, True)[0]
            self.entropy_average(entropy_scalar)
        # Coverage Computation
        if self.coverage_scalar > 0:
            coverage = self._compute_coverage(alpha, mask.float())
            outputs["coverage_loss"] = coverage * self.coverage_scalar
            coverage_scalar = to_numpy(coverage, True)[0]
            self.coverage_average(coverage_scalar)

        outputs["logits"] = logits
        outputs["mask"] = mask
        outputs["log_probs"] = log_probs
        outputs["alpha"] = alpha
        if labels is not None:
            # Compute loss
            smoothing = 0.1
            soft_labels = labels + smoothing - (2 * smoothing * labels)
            loss = -(soft_labels * log_probs + ((1 - soft_labels) * F.logsigmoid(-logits)))
            # non_zero_indices = labels.view(-1).nonzero().view(-1)
            # loss_flat = loss.view(-1)
            # loss = torch.gather(loss_flat, 0, non_zero_indices).sum()
            loss = loss.mean()
            outputs["loss"] = loss
            for key in outputs:
                if re.match(".+_loss$", key) is not None:
                    outputs["loss"] += outputs[key]
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
        if hasattr(self, "entropy_average"):
            metric_dict["entropy_loss"] = self.entropy_average.get_metric(
                reset=reset)
        if hasattr(self, "coverage_average"):
            metric_dict["coverage_loss"] = self.coverage_average.get_metric(
                reset=reset)
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
                text_field_embedder_modules, embedder_type).from_params(
                vocab, embedder_params
            )

        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        attention_params = params.pop("attention_module")
        attention_type = attention_params.pop("type")
        key_dim = attention_params.get("key_emb_size")
        attention_module = getattr(
            attention_modules, attention_type).from_params(attention_params)
        label_namespace = params.pop("label_namespace", "labels")

        dropout = params.pop("dropout", None)
        entropy_scalar = params.pop("entropy_scalar", 0.)
        coverage_scalar = params.pop("coverage_scalar", 0.)
        threshold = params.pop("threshold", 0.5)
        initializer = InitializerApplicator.from_params(
            params.pop('initializer', [])
        )
        regularizer = RegularizerApplicator.from_params(
            params.pop('regularizer', [])
        )
        label_indexer = params.pop("label_indexer", "LabelIndicesBiMap")

        params.assert_empty(cls.__name__)

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   encoder=encoder,
                   attention_module=attention_module,
                   key_dim=key_dim,
                   entropy_scalar=entropy_scalar,
                   coverage_scalar=coverage_scalar,
                   label_namespace=label_namespace,
                   dropout=dropout,
                   thresh=threshold,
                   initializer=initializer,
                   regularizer=regularizer,
                   label_indexer=label_indexer
                   )
