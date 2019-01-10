from typing import Dict, List, Sequence, Iterable
import itertools
import logging

from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.conll2003 import Conll2003DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.fields.multilabel_field import MultiLabelField

@DatasetReader.register("weak_conll2003")
class WeakConll2003DatasetReader(Conll2003DatasetReader):
    """
    Reader for Conll 2003 dataset. Almost identical to AllenNLP's Conll2003DatasetReader, except that it
    returns an additional MultiLabelField that represents which NER tags were present in the sentence in 
    question.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    tag_label: ``str``, optional (default=``ner``)
        Specify `ner`, `pos`, or `chunk` to have that tag loaded into the instance field `tag`.
    feature_labels: ``Sequence[str]``, optional (default=``()``)
        These labels will be loaded as features into the corresponding instance fields:
        ``pos`` -> ``pos_tags``, ``chunk`` -> ``chunk_tags``, ``ner`` -> ``ner_tags``
        Each will have its own namespace: ``pos_tags``, ``chunk_tags``, ``ner_tags``.
        If you want to use one of the tags as a `feature` in your model, it should be
        specified here.
    coding_scheme: ``str``, optional (default=``IOB1``)
        Specifies the coding scheme for ``ner_labels`` and ``chunk_labels``.
        Valid options are ``IOB1`` and ``BIOUL``.  The ``IOB1`` default maintains
        the original IOB1 scheme in the CoNLL 2003 NER data.
        In the IOB1 scheme, I is a token inside a span, O is a token outside
        a span and B is the beginning of span immediately following another
        span of the same type.
    label_namespace: ``str``, optional (default=``labels``)
        Specifies the namespace for the chosen ``tag_label``.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tag_label: str = "ner",
                 feature_labels: Sequence[str] = (),
                 lazy: bool = False,
                 coding_scheme: str = "IOB1",
                 label_namespace: str = "labels") -> None:
        super().__init__(token_indexers, tag_label, feature_labels, lazy, coding_scheme, label_namespace)

    @overrides
    def text_to_instance(self, # type: ignore
                         tokens: List[Token],
                         pos_tags: List[str] = None,
                         chunk_tags: List[str] = None,
                         ner_tags: List[str] = None) -> Instance:
        conll_instance = super().text_to_instance(tokens, pos_tags, chunk_tags, ner_tags)

        # TODO:
        ner_label_list = []

        weak_ner_labels = MultiLabelField(ner_label_list, "weak_ner_labels")
        conll_instance.add_field(weak_ner_labels)
