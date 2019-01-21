from __future__ import absolute_import
import torch
import numpy as np
from torch.nn import Linear
from torch.nn import Dropout
from torch.autograd import Variable
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

import model.attention_module as attention_module
from model.metrics import ClassificationMetrics

from commons.utils import to_numpy

# FIXME: Implement this
from preprocess.label_indices \
    import LabelIndicesBiMap, CondensedLabelIndicesBiMap


class HierAttnNetworkClassifier(Model):
    """Hierarchical Attention Network

    Implements a variant of
    http://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf
    The idea being that the first a word level attention distribution
    is computed for each sentence, and then a sentence level
    attention is computed for the final classification.

    This allows for better segmentation as well as better identification
    of important sentences, useful for coarse chunking.

    Arguments:
        vocab (``Vocabulary``): The vocabulary used for indexing
        text_field_embedder (``TextFieldEmbedder``):
            The text field embedders
        encoder_word (``Seq2SeqEncoder``): Encoding each word in
            a sentence.
        encoder_sent (``Seq2SeqEncoder``): Encoding each sentence
            in a text.
        attn_word (``BaseAttention``): The kind of attention
            being used at a word level
        attn_sent (``BaseAttention``): The kind of attention
            being used at a sentence level
        thresh (float): The class threshold used for predictions.
            (i.e, probs > thresh are classified as 1)
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
                 encoder_word: Seq2SeqEncoder,
                 attn_word: attention_module.BaseAttention,
                 attn_sent: attention_module.BaseAttention,
                 encoder_sent: Seq2SeqEncoder,
                 thresh: float=0.5,
                 label_namespace: str = "labels",
                 dropout: float = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 label_indexer: str = "LabelIndicesBiMap") -> None:
        super(HierAttnNetworkClassifier, self).__init__(vocab, regularizer)

        # Label Information
        self.label_namespace = label_namespace
        self.label_indexer = eval(label_indexer)
        # FIXME: Implement this
        self.num_labels = self.label_indexer.get_num_labels()
        # Prediction thresholds
        self.thresh = thresh
        self.log_thresh = np.log(thresh + 1e-5)

        # Model
        # Text encoders
        self.text_field_embedder = text_field_embedder
        # Sentence and doc encoders
        self.encoder_word = encoder_word
        self.encoder_sent = encoder_sent
        # Attention Modules
        self.key_dim = attn_sent.get_key_dim()
        self.attn_word = attn_word
        self.attn_sent = attn_sent

        if dropout:
            self.dropout = Dropout(dropout)
        else:
            self.dropout = None

        # Label prediction
        self.output_dim = self.attn_sent.get_output_dim()
        self.logits_layer = Linear(self.output_dim, self.num_labels)
        self.classification_metric = ClassificationMetrics(
            self.num_labels, label_indexer)
        initializer(self)

        # Some dimension checks
        check_dimensions_match(
            text_field_embedder.get_output_dim(), encoder_word.get_input_dim(),
            "text field embedding dim", "word encoder input dim")
        check_dimensions_match(
            encoder_word.get_output_dim(), attn_word.get_input_dim(),
            "word encoder output", "word attention input")
        check_dimensions_match(
            attn_word.get_output_dim(), encoder_sent.get_input_dim(),
            "word attention output", "sent encoder input")
        check_dimensions_match(
            encoder_sent.get_output_dim(), attn_sent.get_input_dim(),
            "sent encoder output", "sent attn input")

    def apply_dropout(self, tensor):
        """Apply dropout
        """
        if self.dropout:
            tensor = self.dropout(tensor)
        return tensor

    def _msg2embed(self, sents):
        """Encodes a message (collection of sentences)
        
        Commonly used symbols

            * S : max words per sentence
            * C : max size of each word

        Arguments:
            sents (``Dict[str, LongTensor]``): The sentences.
                contains the following

                    * tokens: num_sents x S: The tokens
                    * chars: num_sents x S x C: The characters
                    * elmo: [optional]: num_sents x S x C : The elmo tokenization

        Returns:
            (``Tensor``, ``Tensor``): sent_encoding, sent_attns

                * sent_encoding: num_sents x embed_dim
                * sent_attns: num_sents x S

        """
        # word -> sents
        emb_msg = self.text_field_embedder(sents)
        word_mask = util.get_text_field_mask(sents)  # num_sents x S
        emb_msg = self.apply_dropout(emb_msg)
        encoded_msg = self.encoder_word(emb_msg, word_mask)
        # num_sents x S x emb_dim
        sent_encoding, sent_attns = self.attn_word(encoded_msg, word_mask)
        # sent_encoding : num_sents x emb_dim
        # sent_attns : num_sents x S
        return sent_encoding, sent_attns

    @overrides
    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """Compute the forward pass

        Commonly refered symbols

            * T : max_seq_len
            * C : max_char_len
            * S : max_sent_len
            * L : num_labels
            * out_dim : The output dim from the attention layer
            * alpha: The attention weights
            * batch : batch size


        Arguments:
            tokens (Dict[str, List[Dict[str, ``LongTensors``]]]):
                Contains the following:

                    * tokens (Unused)
                    * sents :

                        * tokens: num_sents x S
                        * chars: num_sents x S x C
                        * elmo [Optional]: num_sents x S x C

            labels (``LongTensor``) : batch x L: The labels

        Returns:
            Dict[str, ``LongTensor``]: A dictionary with the following
            attributes

                * loss: 1 x 1 : The BCE Loss
                * attentions: List[Dict[str, Torch.LongTensor]]
                  The word and sentence level attentions
                * preds: batch x L : The probabilites predicted
                  Note that each is an independant prediction for the class
                  i.e The columns do **not** sum to 1


        """
        # Write placeholders
        is_cuda = tokens["sents"][0]["tokens"].is_cuda
        batch_size = len(tokens["sents"])
        # placeholders
        lengths = [sent["tokens"].size(0) for sent in tokens["sents"]]
        max_length = max(lengths)
        batched_sent_embeds = Variable(torch.zeros(
            batch_size, max_length, self.encoder_sent.get_input_dim()))
        mask = Variable(torch.zeros(batch_size, max_length))
        attentions = {"word_level": [], "sent_level": None}
        if is_cuda:
            batched_sent_embeds = batched_sent_embeds.cuda()
            mask = mask.cuda()
        for batch_ix in range(batch_size):
            sent_embeds, word_level_attn = self._msg2embed(
                tokens["sents"][batch_ix])
            attentions["word_level"].append(word_level_attn)
            batched_sent_embeds[batch_ix, :lengths[batch_ix], :] = sent_embeds
            mask[batch_ix, :lengths[batch_ix]] = 1.
            # attentions.append(msg_attn)
        # sent -> msg embeds
        encoded_msgs = self.encoder_sent(batched_sent_embeds, mask)
        '''
            encoded_msgs : batch x max_sent_len x output_dim
        '''
        msg_embeds, sent_level_attn = self.attn_sent(encoded_msgs, mask)
        attentions["sent_level"] = sent_level_attn
        '''
            msg_embeds : batch x embed_size
        '''

        logits = self.logits_layer(msg_embeds)  # batch x L
        log_probs = F.logsigmoid(logits)
        outputs = {
            "logits": logits,
            "log_probs": log_probs,
            "attentions": attentions,
            "mask": mask
        }
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
             if pred_labels[batch_ix][label_ix] == 1]  # FIXME Check this
            for batch_ix in range(batch)
        ]
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = self.classification_metric.get_metric(reset=reset)
        return OrderedDict({x: y for x, y in metric_dict.items()})

    @classmethod
    def from_params(
        cls, vocab: Vocabulary,
        params: Params
    ) -> 'HierAttnNetworkClassifier':
        """Constructs class from ``Params`` and ``Vocabulary``
        """
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(
            vocab, embedder_params
        )
        encoder_word = Seq2SeqEncoder.from_params(params.pop("encoder_word"))
        encoder_sent = Seq2SeqEncoder.from_params(params.pop("encoder_sent"))
        attention_word_params = params.pop("attention_word")
        attention_type = attention_word_params.pop("type")
        attn_word = getattr(
            attention_module, attention_type).from_params(
            attention_word_params)

        attention_sent_params = params.pop("attention_sent")
        attention_type = attention_sent_params.pop("type")
        attn_sent = getattr(
            attention_module, attention_type).from_params(
            attention_sent_params)

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
            encoder_word=encoder_word,
            attn_word=attn_word,
            attn_sent=attn_sent,
            encoder_sent=encoder_sent,
            thresh=threshold,
            label_namespace=label_namespace,
            dropout=dropout,
            initializer=initializer,
            regularizer=regularizer,
            label_indexer=label_indexer
        )
