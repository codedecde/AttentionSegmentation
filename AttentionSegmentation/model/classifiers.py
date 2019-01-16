from __future__ import absolute_import
from overrides import overrides
from collections import OrderedDict
from typing import Optional, Dict
import numpy as np
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch import Tensor, LongTensor

# import AttentionSegmentation.model as Attns
import AttentionSegmentation.model as Attns
import AttentionSegmentation.allennlp.nn.util as util
from AttentionSegmentation.allennlp.nn import \
    InitializerApplicator, RegularizerApplicator
from AttentionSegmentation.allennlp.modules import \
    Seq2SeqEncoder, TextFieldEmbedder
from AttentionSegmentation.allennlp.data import Vocabulary
from AttentionSegmentation.allennlp.common.checks import check_dimensions_match
from AttentionSegmentation.allennlp.models.model import Model
from AttentionSegmentation.allennlp.common.params import Params
from AttentionSegmentation.allennlp.training.metrics import \
    BooleanAccuracy

from AttentionSegmentation.reader.label_indexer import LabelIndexer


class Classifier(Model):
    """
        This model computes the predictions, losses, attentions
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        encoder_word: Seq2SeqEncoder,
        attn_word: Attns.BaseAttention,
        label_indexer: LabelIndexer,
        thresh: float = 0.5,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None
    ) -> 'Classifier':
        super(Classifier, self).__init__(vocab, regularizer)
        # Label info
        self.label_indexer = label_indexer
        self.num_labels = self.label_indexer.get_num_tags()
        if self.num_labels != 2:
            raise NotImplementedError(
                "Multi Label Classifiction needs to be handled")
        else:
            # O label is 1 - label
            self.num_labels -= 1
        # Prediction thresholds
        self.thresh = thresh
        self.log_thresh = np.log(thresh + 1e-5)
        # Model
        # Text encoders
        self.text_field_embedder = text_field_embedder
        # Sentence and doc encoders
        self.encoder_word = encoder_word
        # Attention Modules
        self.attn_word = attn_word
        # Label prediction
        self.output_dim = self.attn_word.get_output_dim()
        self.logits_layer = Linear(self.output_dim, self.num_labels)
        # self.classification_metric = ClassificationMetrics(
        #     self.num_labels, label_indexer)
        self.classification_metric = BooleanAccuracy()
        initializer(self)

        # Some dimension checks
        check_dimensions_match(
            text_field_embedder.get_output_dim(), encoder_word.get_input_dim(),
            "text field embedding dim", "word encoder input dim")
        check_dimensions_match(
            encoder_word.get_output_dim(), attn_word.get_input_dim(),
            "word encoder output", "word attention input")

    @overrides
    def forward(
        self,
        tokens: Dict[str, LongTensor],
        labels: LongTensor = None,
        tags: LongTensor = None,
        **kwargs
    ) -> Dict[str, Tensor]:
        """The forward pass

        Commonly used symbols:
            S : Max sent length
            C : Max word length
            L : Number of tags (Including the O tag)

        Arguments:
            tokens (Dict[str, ``LongTensor``]): The indexed values
                Contains the following:
                    * tokens: batch_size x S
                    * chars: batch_size x S x C
                    * elmo [Optional]: batch_size x S x C

            labels (``LongTensor``) : batch x L: The labels
            tags (``LongTensor``) : batch x S : The gold NER tags

        ..note::
            Need to incorporate pos_tags etc. into kwargs

        Returns:
            Dict[str, ``LongTensor``]: A dictionary with the following
            attributes

                * loss: 1 x 1 : The BCE Loss
                * logits: (batch,) : The output of the logits
                    for class prediction
                * log_probs: (batch,) : The output of the F.logsigmoid(logits)
                    for class prediction
                * attentions: batch x S:
                  The attention over each word in the sentence
                * preds: (batch,) : The probabilites predicted
        """
        if len(kwargs) > 0:
            raise NotImplementedError("Don't handle features yet")
        emb_msg = self.text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)  # num_sents x S
        encoded_msg = self.encoder_word(emb_msg, mask)
        sent_emb, sent_attns = self.attn_word(encoded_msg, mask)
        '''
            sent_emb: batch x embed_dim
            sent_attns: batch x T
        '''
        # FIXME: Doing binary classification right now
        logits = self.logits_layer(sent_emb).view(-1)  # batch,
        log_probs = F.logsigmoid(logits)
        outputs = {
            "logits": logits,
            "log_probs": log_probs,
            "attentions": sent_attns,
            "mask": mask
        }
        pred_labels = log_probs.gt(self.log_thresh).long()
        outputs["preds"] = pred_labels
        if labels is not None:
            # Binary classification
            smoothing = 0.1
            # FIXME: Doing binary classification right now
            labels = labels[:, 0].contiguous()
            soft_labels = labels + smoothing - (2 * smoothing * labels)
            loss = -(soft_labels * log_probs +
                     ((1 - soft_labels) * F.logsigmoid(-logits)))
            loss = loss.mean()
            outputs["loss"] = loss
            self.classification_metric(pred_labels, labels.long())

        return outputs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = OrderedDict()
        metric_dict["accuracy"] = self.classification_metric.get_metric(
            reset=reset)
        # return OrderedDict({x: y for x, y in metric_dict.items()})
        return metric_dict

    @classmethod
    @overrides
    def from_params(
        cls,
        vocab: Vocabulary,
        params: Params,
        label_indexer: LabelIndexer
    ) -> 'Classifer':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(
            vocab, embedder_params
        )
        encoder_word = Seq2SeqEncoder.from_params(params.pop("encoder_word"))
        attn_word_params = params.pop("attention_word")
        attn_type = attn_word_params.pop("type")
        attn_word = getattr(Attns, attn_type).from_params(attn_word_params)
        threshold = params.pop("threshold", 0.5)
        initializer = InitializerApplicator.from_params(
            params.pop('initializer', [])
        )
        regularizer = RegularizerApplicator.from_params(
            params.pop('regularizer', [])
        )
        return cls(
            vocab=vocab,
            text_field_embedder=text_field_embedder,
            encoder_word=encoder_word,
            attn_word=attn_word,
            thresh=threshold,
            initializer=initializer,
            regularizer=regularizer,
            label_indexer=label_indexer
        )
