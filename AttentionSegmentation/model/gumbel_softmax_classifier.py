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
import logging
import pdb

# import AttentionSegmentation.model as Attns
import allennlp.nn.util as util
from allennlp.nn import \
    InitializerApplicator, RegularizerApplicator
from allennlp.modules import \
    Seq2SeqEncoder, TextFieldEmbedder
from allennlp.data import Vocabulary
from allennlp.common.checks import check_dimensions_match
from allennlp.modules import FeedForward
from allennlp.models.model import Model
from allennlp.common.params import Params, Registrable
from allennlp.training.metrics import \
    BooleanAccuracy

from AttentionSegmentation.reader.label_indexer import LabelIndexer
from AttentionSegmentation.commons.utils import to_numpy
from AttentionSegmentation.model.metrics import ClassificationMetrics
from AttentionSegmentation.model.base_classifier import BaseClassifier

logger = logging.getLogger(__name__)


class Generator(torch.nn.Module, Registrable):
    def __init__(self, *args, **kwargs):
        super(Generator, self).__init__()

    @classmethod
    def from_params(cls, params: Params):
        gen_type = params.pop("type")
        return cls.by_name(gen_type).from_params(params)


@Generator.register("basic_generator")
class BasicGenerator(Generator):
    def __init__(
        self,
        encoder_word: Seq2SeqEncoder,
        prob_layer: FeedForward
    ):
        self.encoder_word = encoder_word
        self.prob_layer = prob_layer
        super(BasicGenerator, self).__init__()

    @overrides
    def forward(
        self,
        emb_msg: torch.Tensor,
        mask: torch.LongTensor
    ) -> torch.Tensor:
        logits = self.encoder_word(emb_msg, mask)
        attentions = self.prob_layer(logits)
        return attentions

    @classmethod
    def from_params(cls, params: Params):
        encoder_word = Seq2SeqEncoder.from_params(params.pop("encoder_word"))
        prob_layer = FeedForward.from_params(params.pop("prob_layer"))
        assert params.assert_empty(cls.__name__)
        return cls(
            encoder_word=encoder_word,
            prob_layer=prob_layer
        )


class Sampler(torch.nn.Module, Registrable):
    def __init__(self, *args, **kwargs):
        super(Sampler, self).__init__()

    @classmethod
    def from_params(cls, params: Params):
        sampler_type = params.pop("type")
        return cls.by_name(sampler_type).from_params(params)


@Sampler.register("identity")
class Identity(Sampler):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    @overrides
    def forward(
        self,
        attentions: torch.Tensor,
        mask: torch.LongTensor
    )->'Identity':
        """Generates the probability. Identity function.
        Except applies the mask

        Parameters:
            attentions (torch.Tensor): batch x seq_len x (num_tags + 1)
            mask (torch.LongTensor): batch x seq_len
        Returns:
            samples (torch.Tensor): batch x seq_len x (num_tags + 1)
        """
        mask = mask.float()
        samples = mask.unsqueeze(-1).expand_as(attentions) * attentions
        return samples

    @classmethod
    def from_params(cls, params: Params):
        params.assert_empty(cls.__name__)
        return cls()


class Identifier(torch.nn.Module, Registrable):
    def __init__(self, *args, **kwargs):
        super(Identifier, self).__init__()

    @classmethod
    def from_params(cls, params: Params, label_indexer: LabelIndexer):
        identifier_type = params.pop("type")
        return cls.by_name(identifier_type).from_params(params, label_indexer)


@Identifier.register("basic_identifier")
class BasicIdentifier(Identifier):
    def __init__(
        self,
        label_indexer: LabelIndexer,
        thresh: float,
        output_dim: int
    ):
        self.label_indexer = label_indexer
        self.thresh = thresh
        self.output_dim
        self.log_thresh = np.log(self.thresh + 1e-5)
        num_tags = self.label_indexer.get_num_tags()
        for tag_ix in range(num_tags):
            tag = self.label_indexer.get_tag(tag_ix)
            module = Linear(self.output_dim, 1)
            setattr(self, f"logits_layer{tag}", module)

        super(BasicIdentifier, self).__init__()

    def forward(
        self,
        emb_msg: torch.Tensor,
        mask: torch.LongTensor,
        samples: torch.Tensor,
        labels: torch.LongTensor
    )->torch.Tensor:
        """
        Parameters:
            emb_msg: batch x seq_len x embed_dim
            mask: batch x seq_len
            samples: batch x seq_len x (num_tags + 1)
            labels: batch
        """
        outputs = {}
        outputs["mask"] = mask
        batch, seq_len, embed_dim = emb_msg.size()
        batch, seq_len, num_tags_with_other = samples.size()
        num_tags = num_tags_with_other - 1
        # Now generate the sentence embeddings
        all_logits = []
        for tag_ix in range(num_tags):
            tag = self.label_indexer.get_tag(tag_ix)
            sentence_embed = \
                emb_msg * mask.float().unsqueeze(-1).expand_as(emb_msg) * \
                samples[:, :, tag_ix].unsqueeze(-1).expand_as(emb_msg)
            sentence_embed = sentence_embed.sum(dim=1)
            assert sentence_embed.size() == (batch, embed_dim)
            logits = getattr(self, f"logits_layer_{tag}")(sentence_embed)
            all_logits.append(logits)
        all_logits = torch.cat(all_logits, -1)
        log_probs = F.logsigmoid(all_logits)
        outputs["logits"] = all_logits
        outputs["log_probs"] = log_probs
        pred_labels = log_probs.gt(self.log_thresh).long()
        outputs["preds"] = pred_labels
        if labels is not None:
            labels = labels[:, :-1]
            soft_labels = labels + self.smoothing - \
                (2 * self.smoothing * labels)
            loss = -(soft_labels * log_probs +
                     ((1 - soft_labels) * F.logsigmoid(-all_logits)))
            loss = loss.mean(-1).mean()
            outputs["loss"] = loss
        return outputs

    @classmethod
    def from_params(cls, params: Params, label_indexer: LabelIndexer):
        thresh = params.pop_float("thresh", 0.5)
        output_dim = params.pop_int("output_dim")
        params.assert_empty(cls.__name__)
        return cls(
            label_indexer=label_indexer,
            thresh=thresh,
            output_dim=output_dim
        )


@BaseClassifier.register("recurrent_attention_classifier")
class RecurrentAttentionClassifier(BaseClassifier):
    """This class is similar to the previous one, except that
    it handles multi level classification
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        generator: Generator,
        sampler: Sampler,
        identifier: Identifier,
        label_indexer: LabelIndexer,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None
    ) -> 'RecurrentAttentionClassifier':
        super(RecurrentAttentionClassifier, self).__init__(vocab, regularizer)
        # Label info
        self.label_indexer = label_indexer
        self.num_labels = self.label_indexer.get_num_tags()

        # Prediction thresholds
        # self.thresh = thresh
        # self.log_thresh = np.log(thresh + 1e-5)
        # Model
        # Text encoders
        self.text_field_embedder = text_field_embedder
        self.generator = generator

        # Attention Modules
        # We use setattr, so that cuda properties translate.
        self.identifier = identifier

        self.classification_metric = ClassificationMetrics(
            label_indexer)
        # self.classification_metric = BooleanAccuracy()
        initializer(self)

        # Some dimension checks
        # FIXME:
        # Do Dimension Checks
        # check_dimensions_match(
        #     text_field_embedder.get_output_dim(), encoder_word.get_input_dim(),
        #     "text field embedding dim", "word encoder input dim")
        # check_dimensions_match(
        #     encoder_word.get_output_dim(), attn_word[0].get_input_dim(),
        #     "word encoder output", "word attention input")

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
        mask = util.get_text_field_mask(tokens)
        attentions = self.generator(emb_msg, mask)
        samples = self.sampler(attentions, mask)
        outputs = self.identifier(emb_msg, mask, samples, labels)
        outputs["attentions"] = attentions[:, :, :-1]
        pred_labels = outputs["pred"]
        self.classification_metric(pred_labels.long(), labels.long())
        self.decode(outputs)
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
        log_probs = to_numpy(
            outputs["log_probs"], outputs["log_probs"].is_cuda)
        attentions = to_numpy(
            outputs["attentions"], outputs["attentions"].is_cuda)
        for ix in range(lengths.size):
            non_zero_indices = np.nonzero(predictions[ix])[0]
            pred_list = []
            for kx in range(non_zero_indices.shape[0]):
                pred_list.append(
                    [
                        self.label_indexer.get_tag(
                            non_zero_indices[kx]
                        ),
                        np.exp(log_probs[ix, non_zero_indices[kx]])
                    ]
                )

            if len(pred_list) == 0:
                pred_list.append(["O", 1.0])
            decoded_output["preds"].append(pred_list)
            attention = OrderedDict()
            for jx in range(attentions[ix].shape[-1]):
                tag = self.label_indexer.get_tag(jx)
                attention[tag] = attentions[ix, :lengths[ix], jx].tolist()
            decoded_output["attentions"].append(attention)
        return decoded_output

    @classmethod
    @overrides
    def from_params(
        cls,
        vocab: Vocabulary,
        params: Params,
        label_indexer: LabelIndexer
    ) -> 'RecurrentAttentionClassifier':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(
            vocab, embedder_params
        )
        gen_params = params.pop("generator_params")
        generator = Generator.from_params(
            gen_params
        )

        sampler_params = params.pop("sampler_params")
        sampler = Sampler.from_params(
            sampler_params
        )

        identifier_params = params.pop("identifier_params")
        identifier = Identifier.from_params(
            identifier_params,
            label_indexer
        )

        initializer = InitializerApplicator.from_params(
            params.pop('initializer', [])
        )
        regularizer = RegularizerApplicator.from_params(
            params.pop('regularizer', [])
        )
        return cls(
            vocab=vocab,
            text_field_embedder=text_field_embedder,
            generator=generator,
            sampler=sampler,
            identifier=identifier,
            initializer=initializer,
            regularizer=regularizer,
            label_indexer=label_indexer
        )
