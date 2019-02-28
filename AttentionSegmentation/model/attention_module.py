from typing import Tuple
from overrides import overrides
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pdb

from allennlp.common.params import Params


class BaseAttention(nn.Module):
    """An abstract attention module.

    All attention classes inherit from this class.
    Attention refers to focusing on the context based
    on a (usually learnt) key.

    Arguments:
        input_emb_size (int): The ctxt embedding size.
        key_emb_size (int): The size of the key
        output_emb_size (int): The output embedding size
    """

    def __init__(self,
                 input_emb_size: int,
                 key_emb_size: int,
                 output_emb_size: int):
        super(BaseAttention, self).__init__()
        self.input_emb_size = input_emb_size
        self.key_emb_size = key_emb_size
        self.output_emb_size = output_emb_size

    def forward(self, *args, **kwargs):
        raise NotImplementedError("This is an abstract class")

    def get_output_dim(self):
        """The output embedding size
        """
        return self.output_emb_size

    def get_key_dim(self):
        """The key embedding size
        """
        return self.key_emb_size

    def get_input_dim(self):
        """The input embedding size
        """
        return self.input_emb_size


class KeyedAttention(BaseAttention):
    """Computes a single attention distribution,
    based on a learned key.

    Arguments:
        key_dim (int): The size of the key
        ctxt_dim (int): The size of the context
        attn_type (str): ("sum"|"dot") The type of
            attention mechanism. "sum" is the usual
            Bahdanau model of attention, while
            "dot" is the dot product

    """

    def __init__(self, key_dim, ctxt_dim, attn_type, dropout=0.0,
                 temperature=1., use_sentinel=False):
        super(KeyedAttention, self).__init__(
            input_emb_size=ctxt_dim,
            key_emb_size=key_dim,
            output_emb_size=ctxt_dim
        )
        self.attn_type = attn_type
        self.proj_ctxt = nn.Linear(ctxt_dim, key_dim)
        self.key = nn.Parameter(torch.Tensor(key_dim, 1).uniform_(-0.01, 0.01))
        self.use_sentinel = use_sentinel
        if use_sentinel:
            self.sentinel_emb = nn.Parameter(
                torch.Tensor(ctxt_dim).uniform_(-0.01, 0.01))
        self._dropout = nn.Dropout(p=dropout) if dropout != 0. else None
        self.temperature = temperature
        if self.attn_type == "sum":
            self.proj_ctxt_key_matrix = nn.Linear(key_dim, 1)

    def forward(self, context, mask, attns_mask=None):
        """The forward pass

        Arguments:
            context (``torch.Tensor``):
                batch x seq_len x embed_dim: The Context
            mask (``torch.LongTensor``):
                batch x seq_len: The context mask
        Returns:
            (``torch.Tensor``, ``torch.Tensor``)

            weighed_emb: batch x output_embed_size:
            The attention weighted embeddings

            attn_weights: batch x seq_len
            The attention weights

        """
        new_ctxt = None
        new_mask = None
        new_attns_mask = None
        if self.use_sentinel:
            batch, seq_len, ctxt_dim = context.size()
            new_ctxt = torch.zeros(batch, seq_len + 1, ctxt_dim)
            new_ctxt.requires_grad = context.requires_grad
            if context.is_cuda:
                new_ctxt = new_ctxt.cuda()
            new_ctxt[:, :-1, :] = context
            new_ctxt[:, -1, :] = context[:, -1, :]
            if mask is not None:
                new_mask = torch.zeros(batch, seq_len + 1)
                new_mask.requires_grad = mask.requires_grad
                if mask.is_cuda:
                    new_mask = new_mask.cuda()
                new_mask[:, :-1] = mask
                lengths = mask.sum(-1)
                for ix in range(batch):
                    new_ctxt[ix, lengths[ix], :] = self.sentinel_emb
                    new_mask[ix, lengths[ix]] = 1
                # Now the attn masks
                if attns_mask is not None:
                    new_attns_mask = torch.zeros(batch, seq_len + 1)
                    new_attns_mask.requires_grad = attns_mask.requires_grad
                    if attns_mask.is_cuda:
                        new_attns_mask = new_attns_mask.cuda()
                    new_attns_mask[:, :-1] = attns_mask
        else:
            new_ctxt = context
            new_mask = mask
            new_attns_mask = attns_mask
        proj_ctxt = self.proj_ctxt(new_ctxt)  # batch x seq_len x key_dim
        if self.attn_type == "dot":
            batch, seq_len, key_dim = proj_ctxt.size()
            scores = torch.mm(proj_ctxt.view(-1, key_dim), self.key)
            logits = scores.contiguous().view(batch, seq_len)
        elif self.attn_type == "sum":
            batch, seq_len, key_dim = proj_ctxt.size()
            expanded_key = self.key.transpose(0, 1).expand(batch, seq_len, -1)
            ctxt_key_matrix = torch.tanh(expanded_key + proj_ctxt)
            logits = self.proj_ctxt_key_matrix(ctxt_key_matrix).squeeze(-1)
        logits /= self.temperature
        if new_mask is not None:
            float_mask = new_mask.float()
            negval = -10e5
            logits = (float_mask * logits) + ((1 - float_mask) * negval)
            if new_attns_mask is not None:
                # Mask out the masked tokens
                new_attns_mask = new_attns_mask.float()
                logits = (logits * (1. - new_attns_mask)) + (negval * new_attns_mask)
        attn_weights = F.softmax(logits, -1).unsqueeze(1)
        if self._dropout is not None:
            attn_weights = self._dropout(attn_weights)
        weighted_emb = torch.bmm(attn_weights, new_ctxt).squeeze(1)
        return weighted_emb, attn_weights.squeeze(1)

    @classmethod
    def from_params(cls, params: Params):
        """Construct from ``Params``
        """
        key_dim = params.pop("key_emb_size")
        ctxt_emb_size = params.pop("ctxt_emb_size")
        attn_type = params.pop("attn_type")
        dropout = params.pop("dropout", 0.0)
        temperature = params.pop("temperature", 1.0)
        use_sentinel = params.pop("use_sentinel", False)
        params.assert_empty(cls.__name__)
        return cls(
            key_dim=key_dim,
            ctxt_dim=ctxt_emb_size,
            attn_type=attn_type,
            dropout=dropout,
            temperature=temperature,
            use_sentinel=use_sentinel
        )


class DotAttention(BaseAttention):
    """This computes attention values based on
    dot products for scores.

    This class computes num_label attention distributions,
    one for each label, however the projection network share
    parameters. Note that the weight in proj_to_label_namespace
    corresponds to the key for each label.

    Arguments:
        key_emb_size (int): The vector size of the key
        ctxt_emb_size (int): The embedding size of the ctxt
        num_labels (int): The number of labels

    """

    def __init__(self,
                 key_emb_size: int,
                 ctxt_emb_size: int,
                 num_labels: int):
        super(DotAttention, self).__init__(
            input_emb_size=ctxt_emb_size,
            key_emb_size=key_emb_size,
            output_emb_size=key_emb_size
        )
        self.ctxt_emb_size = ctxt_emb_size
        self.key_emb_size = key_emb_size
        self.num_labels = num_labels
        self.proj_to_label_space = nn.Linear(key_emb_size, num_labels)
        self.ctxt_proj = nn.Linear(ctxt_emb_size, key_emb_size)

    @overrides
    def forward(self,
                context: torch.Tensor,
                mask: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """The forward pass.

        * ctxt_emb_size : The context embedding size
        * T : max sequence length
        * L : number of labels

        Arguments:
            context (``torch.Tensor``):
                batch x T x ctxt_emb_size: The context
            mask (``torch.LongTensor``):
                batch x T: The mask for the context

        Returns:
            (``torch.Tensor``, ``torch.Tensor``)

            output_embeds: batch x L x output_embed_size:
            The attention weighted embeddings

            alphas: batch x L x T:
            The attention weights

        """
        encoded_ctxt = self.ctxt_proj(context)  # batch x T x hdim
        batch, max_seq_len, hdim = encoded_ctxt.size()
        logits = self.proj_to_label_space(encoded_ctxt)  # batch x T x L
        negval = -10e5
        float_mask = mask.unsqueeze(-1).expand(-1, -1, logits.size(-1)).float()
        logits = (float_mask * logits) + (negval * (1. - float_mask))  # batch X T x L
        alpha = F.softmax(logits, 1).transpose(1, 2)  # batch x L x T
        batch, num_labels, max_seq_len = alpha.size()
        expanded_ctxt = encoded_ctxt.unsqueeze(1).expand(
            -1, num_labels, -1, -1).contiguous().view(
            -1, max_seq_len, hdim)  # batch * L x T x hdom
        alpha_flat = alpha.contiguous().view(
            -1, max_seq_len).unsqueeze(1)  # batch * L x 1 x T
        weighted_ctxt = torch.bmm(
            alpha_flat, expanded_ctxt).squeeze(1)  # batch * L x hdim
        outputs = weighted_ctxt.view(batch, num_labels, -1)  # batch x L x hdim
        return outputs, alpha

    @classmethod
    def from_params(cls, params: Params) -> 'DotAttention':
        """Constructs class from ``Params``
        """
        key_emb_size = params.pop("key_emb_size")
        ctxt_emb_size = params.pop("ctxt_emb_size")
        num_labels = params.pop("num_labels")
        params.assert_empty(cls.__name__)
        return DotAttention(
            key_emb_size=key_emb_size,
            ctxt_emb_size=ctxt_emb_size,
            num_labels=num_labels
        )


class Ctxt2Attn(nn.Module):
    """A self attention layer.

    In this case, we don't need a key for the attention,
    rather, like most self attention methods, it simply
    projects the context, computes the scores, sharing parameters

    Arguments:
        ctxt_dim (int): The context dimension
        hidden_dim (int): The hidden dimension
            before computing the self attention
        activation (str): The activation function used
            for the self attention (default="Tanh")

    """

    def __init__(self,
                 ctxt_dim,
                 hidden_dim,
                 activation='Tanh'):
        super(Ctxt2Attn, self).__init__()
        self.ctxt_dim = ctxt_dim
        self.hidden_dim = hidden_dim
        self.activation = getattr(nn, activation)()
        self.ctxt_proj_layer = nn.Linear(ctxt_dim, hidden_dim)
        self.scores_proj_layer = nn.Linear(hidden_dim, 1)

    def forward(self, ctxt, mask):
        """The forward pass.

        Arguments:
            ctxt (``torch.Tensor``):
                batch x T x ctxt_dim : The Context
            mask (``torch.LongTensor``):
                batch x T : The Context mask
        Returns:
            (``torch.Tensor``, ``torch.Tensor``)

            output: batch x ctxt_dim:
            The weighted output

            alpha: batch x T :
            The attention weights
        """
        proj_ctxt = self.activation(
            self.ctxt_proj_layer(ctxt))  # batch x T x hidden_dim
        scores = self.scores_proj_layer(proj_ctxt).squeeze(-1)  # batch X T
        negval = -10e5
        float_mask = mask.float()
        scores = (scores * float_mask) + ((1 - float_mask) * negval)
        alpha = F.softmax(scores, -1)
        outputs = torch.bmm(
            alpha.unsqueeze(1), ctxt).squeeze(1)  # batch x ctxt_dim
        return outputs, alpha


class BahdanauAttention(BaseAttention):
    """This computes attention values based on
    the original attention method proposed by
    Bahdanau et. al (https://arxiv.org/abs/1409.0473)

    This class computes num_label attention distributions,
    one for each label, with projection networks also separate.

    Arguments:
        key_emb_size (int): The vector size of the key
        ctxt_emb_size (int): The embedding size of the ctxt
        num_labels (int): The number of labels

    """

    def __init__(self,
                 key_emb_size: int,
                 ctxt_emb_size: int,
                 num_labels: int):
        super(BahdanauAttention, self).__init__(
            input_emb_size=ctxt_emb_size,
            key_emb_size=key_emb_size,
            output_emb_size=ctxt_emb_size
        )
        self.ctxt_emb_size = ctxt_emb_size
        self.key_emb_size = key_emb_size
        self.num_labels = num_labels
        for ix in range(num_labels):
            attn_module = Ctxt2Attn(ctxt_emb_size, key_emb_size)
            '''
                Using a list does not propagate cuda(), causing
                annoying issues, if cuda*() is not passed.
                Setting the module attr dynamically
                achieves the required result, albeit in an
                ugly way.
            '''
            setattr(self, f"attn_module_{ix}", attn_module)

    @overrides
    def forward(self,
                context: torch.Tensor,
                mask: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """The forward pass.

        Check the paper for actual implementation details.
        On a high level, involves adding the key to a projected
        context, passing it through a MLP to compute scores, and
        returning the context, weighed by the softmax scores.

        Arguments:
            context (``torch.Tensor``):
                batch x T x ctxt_emb_size: The context
            mask (``torch.LongTensor``):
                batch x T: The mask for the context
        Returns:
            (``torch.Tensor``, ``torch.Tensor``)

            output_embeds: batch x L x output_embed_size:
            The attention weighted embeddings

            alphas: batch x L x T:
            The attention weights

        """
        batch, max_seq_len, ctxt_dim = context.size()
        is_cuda = context.is_cuda
        # placeholders
        outputs = Variable(torch.zeros(batch, self.num_labels, ctxt_dim))
        alphas = Variable(torch.zeros(batch, self.num_labels, max_seq_len))
        if is_cuda:
            outputs = outputs.cuda()
            alphas = alphas.cuda()

        for ix in range(self.num_labels):
            output, alpha = getattr(self, f"attn_module_{ix}")(context, mask)

            outputs[:, ix, :] = output
            alphas[:, ix, :] = alpha

        return outputs, alphas

    @classmethod
    def from_params(cls, params: Params) -> 'BahdanauAttention':
        """Constructs class from ``Params``
        """
        key_emb_size = params.pop("key_emb_size")
        ctxt_emb_size = params.pop("ctxt_emb_size")
        num_labels = params.pop("num_labels")
        params.assert_empty(cls.__name__)
        return BahdanauAttention(
            key_emb_size=key_emb_size,
            ctxt_emb_size=ctxt_emb_size,
            num_labels=num_labels
        )
