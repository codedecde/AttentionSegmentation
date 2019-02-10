from __future__ import absolute_import
from overrides import overrides
from collections import OrderedDict
from typing import Optional, Dict, List
import numpy as np
import torch
from copy import deepcopy
from torch.nn import Linear
import torch.nn.functional as F
from torch import Tensor, LongTensor
import pdb

# import AttentionSegmentation.model as Attns
import allennlp.nn.util as util
from allennlp.nn import \
    InitializerApplicator, RegularizerApplicator
from allennlp.modules import \
    Seq2SeqEncoder, TextFieldEmbedder
from allennlp.data import Vocabulary
from allennlp.common.checks import check_dimensions_match
from allennlp.models.model import Model
from allennlp.common.params import Params
from allennlp.training.metrics import \
    BooleanAccuracy

from AttentionSegmentation.reader.label_indexer import LabelIndexer
from AttentionSegmentation.commons.utils import to_numpy
import AttentionSegmentation.model as Attns
from AttentionSegmentation.model.metrics import ClassificationMetrics


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

    @overrides
    def decode(self, outputs):
        """
        This decodes the outputs of the model into a format used downstream
        for predictions

        Arguments:
            outputs (List[Dict]) : The outputs generated by the model
                Must contain
                    * mask : The mask for the current batch
                    * preds (batch x num_tags - 1) : The predictions
                        note that if nothing is predicted, we predict "O"
                    * attentions (batch x seq_len x num_tags) : The attentions

        Returns:
            decoded_output (Dict) : The decoded output
                Must contain:
                    * preds (List[List[str]]) : The predicted tags
                    * attentions (List[Dict]) : List of dictionaries
                        mapping each tag to its attention distribution
        """
        decoded_output = {
            "preds": [],
            "attentions": []
        }
        lengths = outputs["mask"].sum(-1)
        tag = self.label_indexer.ix2tags[0]
        lengths = to_numpy(lengths, lengths.is_cuda)
        for ix in range(lengths.size):
            is_cuda = outputs["preds"][ix].is_cuda
            pred = to_numpy(outputs["preds"][ix], is_cuda).item()
            pred_str = tag if pred == 1 else 'O'
            decoded_output["preds"].append([pred_str])
            attention = to_numpy(
                outputs["attentions"][ix, :lengths[ix]], is_cuda).tolist()
            decoded_output["attentions"].append({tag: attention})
        return decoded_output

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


class MultiClassifier(Model):
    """This class is similar to the previous one, except that
    it handles multi level classification
    """

    def __init__(
        self,
        vocab: Vocabulary,
        method: str,
        text_field_embedder: TextFieldEmbedder,
        encoder_word: Seq2SeqEncoder,
        attn_word: List[Attns.BaseAttention],
        label_indexer: LabelIndexer,
        thresh: float = 0.5,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None
    ) -> 'MultiClassifier':
        super(MultiClassifier, self).__init__(vocab, regularizer)
        # Label info
        self.label_indexer = label_indexer
        self.num_labels = self.label_indexer.get_num_tags()
        self.method = method

        # Prediction thresholds
        self.thresh = thresh
        self.log_thresh = np.log(thresh + 1e-5)
        # Model
        # Text encoders
        self.text_field_embedder = text_field_embedder
        # Sentence and doc encoders
        self.encoder_word = encoder_word
        # Attention Modules
        # We use setattr, so that cuda properties translate.
        # Otherwise, it becomes a little messy
        for ix in range(self.num_labels - 1):
            tag = self.label_indexer.get_tag(ix)
            setattr(self, f"attn_{tag}", attn_word[ix])
        # Label prediction
        self.output_dim = attn_word[0].get_output_dim()
        if self.method == "binary":
            for ix in range(self.num_labels):
                tag = self.label_indexer.get_tag(ix)
                module = Linear(self.output_dim, 1)
                setattr(self, f"logits_layer_{tag}", module)
        elif self.method == "softmax":
            module = Linear(
                self.output_dim * (self.num_labels - 1), self.num_labels)
            setattr(self, "logits_layer", module)
        else:
            raise NotImplementedError("Not implemented")

        self.classification_metric = ClassificationMetrics(
            label_indexer)
        # self.classification_metric = BooleanAccuracy()
        initializer(self)

        # Some dimension checks
        check_dimensions_match(
            text_field_embedder.get_output_dim(), encoder_word.get_input_dim(),
            "text field embedding dim", "word encoder input dim")
        check_dimensions_match(
            encoder_word.get_output_dim(), attn_word[0].get_input_dim(),
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
                * logits: (batch, num_tags) : The output of the logits
                    for class prediction
                * log_probs: (batch, num_tags) : The output
                    for class prediction
                * attentions: List[batch x S]:
                  The attention over each word in the sentence,
                  for each tag
                * preds: (batch, num_tags) : The probabilites predicted
        """
        if len(kwargs) > 0:
            raise NotImplementedError("Don't handle features yet")
        emb_msg = self.text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)  # num_sents x S
        encoded_msg = self.encoder_word(emb_msg, mask)
        attentions = []
        sent_embs = []
        for ix in range(self.num_labels - 1):
            tag = self.label_indexer.get_tag(ix)
            sent_emb, sent_attn = getattr(self, f"attn_{tag}")(
                encoded_msg, mask)
            '''
                sent_emb: batch x embed_dim
                sent_attns: batch x T
            '''
            sent_embs.append(sent_emb)
            attentions.append(sent_attn.unsqueeze(-1))

        attentions = torch.cat(attentions, -1)

        outputs = {
            "attentions": attentions,
            "mask": mask
        }

        smoothing = 0.1
        if self.method == "binary":
            all_logits = []
            for ix in range(self.num_labels - 1):
                tag = self.label_indexer.get_tag(ix)
                sent_emb = sent_embs[ix]
                logits = getattr(self, f"logits_layer_{tag}")(sent_emb)
                all_logits.append(logits)
            all_logits = torch.cat(all_logits, -1)
            log_probs = F.logsigmoid(all_logits)
            outputs["logits"] = all_logits
            outputs["log_probs"] = log_probs
            pred_labels = log_probs.gt(self.log_thresh).long()
            outputs["preds"] = pred_labels
            if labels is not None:
                labels = labels[:, :-1]
                soft_labels = labels + smoothing - (2 * smoothing * labels)
                loss = -(soft_labels * log_probs +
                         ((1 - soft_labels) * F.logsigmoid(-all_logits)))
                loss = loss.mean(-1).mean()
                outputs["loss"] = loss
                self.classification_metric(pred_labels.long(), labels.long())
                self.decode(outputs)
        else:
            # softmax method
            sent_emb = torch.cat([sent_embs], -1)
            all_logits = self.logits_layer(sent_emb)
            raise NotImplementedError("Work in progress")
        return outputs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = self.classification_metric.get_metric(
            reset=reset)
        # return OrderedDict({x: y for x, y in metric_dict.items()})
        return metric_dict

    @overrides
    def decode(self, outputs):
        """
        This decodes the outputs of the model into a format used downstream
        for predictions

        Arguments:
            outputs (List[Dict]) : The outputs generated by the model
                Must contain
                    * mask : The mask for the current batch
                    * preds (batch x num_tags - 1) : The predictions
                        note that if nothing is predicted, we predict "O"
                    * attentions (batch x seq_len x num_tags) : The attentions

        Returns:
            decoded_output (Dict) : The decoded output
                Must contain:
                    * preds (List[List[str]]) : The predicted tags
                    * attentions (List[Dict]) : List of dictionaries
                        mapping each tag to its attention distribution
        """
        decoded_output = {
            "preds": [],
            "attentions": []
        }
        lengths = outputs["mask"].sum(-1)
        lengths = to_numpy(lengths, lengths.is_cuda)
        predictions = to_numpy(outputs["preds"], outputs["preds"].is_cuda)
        attentions = to_numpy(
            outputs["attentions"], outputs["attentions"].is_cuda)
        for ix in range(lengths.size):
            non_zero_indices = np.nonzero(predictions[ix])[0]
            pred_list = []
            for ix in range(non_zero_indices.shape[0]):
                pred_list.append(
                    self.label_indexer.get_tag(
                        non_zero_indices[ix]
                    )
                )

            if len(pred_list) == 0:
                pred_list.append("O")
            decoded_output["preds"].append(pred_list)
            attention = OrderedDict()
            for jx in range(attentions[ix].shape[-1]):
                tag = self.label_indexer.get_tag(jx)
                attention[tag] = attentions[ix, :lengths[ix], jx]
            decoded_output["attentions"].append(attention)
        return decoded_output

    @classmethod
    @overrides
    def from_params(
        cls,
        vocab: Vocabulary,
        params: Params,
        label_indexer: LabelIndexer
    ) -> 'MultiClassifer':
        method = params.pop("method", "binary")
        num_tags = label_indexer.get_num_tags()
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(
            vocab, embedder_params
        )
        encoder_word = Seq2SeqEncoder.from_params(params.pop("encoder_word"))
        attn_word_params = params.pop("attention_word")
        attn_type = attn_word_params.pop("type")

        attn_word = []
        for ix in range(num_tags - 1):
            # Since from_params clears out the dictionaries,
            # this deepcopy is necessary
            tmp_attn_params = deepcopy(attn_word_params)
            attn_word.append(
                getattr(Attns, attn_type).from_params(tmp_attn_params))

        threshold = params.pop("threshold", 0.5)
        initializer = InitializerApplicator.from_params(
            params.pop('initializer', [])
        )
        regularizer = RegularizerApplicator.from_params(
            params.pop('regularizer', [])
        )
        return cls(
            vocab=vocab,
            method=method,
            text_field_embedder=text_field_embedder,
            encoder_word=encoder_word,
            attn_word=attn_word,
            thresh=threshold,
            initializer=initializer,
            regularizer=regularizer,
            label_indexer=label_indexer
        )
