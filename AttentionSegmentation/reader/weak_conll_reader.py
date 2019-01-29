from typing import Dict, List, Sequence, Iterable
import itertools
import logging
import logging
import re

from overrides import overrides

from allennlp.common import Params
from allennlp.data.dataset_readers.dataset_reader \
    import DatasetReader
from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers \
    import Conll2003DatasetReader, DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.fields \
    import MultiLabelField, TextField, SequenceLabelField
from AttentionSegmentation.reader.label_indexer import LabelIndexer


logger = logging.getLogger(__name__)
NUM_TOKEN = "@@NUM@@"


def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ''
    if empty_line:
        return True
    else:
        first_token = line.split()[0]
        if first_token == "-DOCSTART-":  # pylint: disable=simplifiable-if-statement
            return True
        else:
            return False


_VALID_LABELS = {'ner', 'pos', 'chunk'}


class WeakConll2003DatasetReader(DatasetReader):
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
                 convert_numbers: bool = False,
                 coding_scheme: str = "IOB1",
                 label_indexer: LabelIndexer = None) -> None:
        super(WeakConll2003DatasetReader, self).__init__(lazy)
        self._token_indexers = token_indexers or {
            'tokens': SingleIdTokenIndexer()}
        if tag_label is not None and tag_label not in _VALID_LABELS:
            raise ConfigurationError("unknown tag label type: {}".format(
                tag_label))
        for label in feature_labels:
            if label not in _VALID_LABELS:
                raise ConfigurationError(
                    "unknown feature label type: {}".format(label))
        if coding_scheme not in ("IOB1", "BIOUL"):
            raise ConfigurationError(
                "unknown coding_scheme: {}".format(coding_scheme))

        self.tag_label = tag_label
        self.feature_labels = set(feature_labels)
        self.coding_scheme = coding_scheme

        self.label_indexer = label_indexer
        self.convert_numbers = convert_numbers

    def get_label_indexer(self):
        return self.label_indexer

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        with open(file_path, "r") as data_file:
            logger.info(
                "Reading instances from lines in file at: %s", file_path)

            # Group into alternative divider / sentence chunks.
            for is_divider, lines in itertools.groupby(data_file, _is_divider):
                # Ignore the divider chunks, so that `lines`
                # corresponds to the words of a single sentence.
                if not is_divider:
                    fields = [line.strip().split() for line in lines]
                    # unzipping trick returns tuples, but our Fields need lists
                    tokens, pos_tags, chunk_tags, ner_tags = [
                        list(field) for field in zip(*fields)
                    ]
                    # TextField requires ``Token`` objects
                    new_tokens = []
                    for token in tokens:
                        if self.convert_numbers:
                            token = re.sub(r"[0-9]+", NUM_TOKEN, token)
                            # if re.match(r"^[0-9]+$", token):
                            #     token = NUM_TOKEN
                        new_tokens.append(Token(token))
                    # tokens = [Token(token) for token in tokens]
                    sequence = TextField(new_tokens, self._token_indexers)

                    instance_fields: Dict[str, Field] = {'tokens': sequence}

                    # Recode the labels if necessary.
                    if self.coding_scheme == "BIOUL":
                        coded_chunks = iob1_to_bioul(chunk_tags)
                        coded_ner = iob1_to_bioul(ner_tags)
                    else:
                        # the default IOB1
                        coded_chunks = chunk_tags
                        coded_ner = ner_tags

                    # Add "feature labels" to instance
                    if 'pos' in self.feature_labels:
                        instance_fields['pos_tags'] = SequenceLabelField(
                            pos_tags, sequence, "pos_tags")
                    if 'chunk' in self.feature_labels:
                        instance_fields['chunk_tags'] = SequenceLabelField(
                            coded_chunks, sequence, "chunk_tags")
                    if 'ner' in self.feature_labels:
                        instance_fields['ner_tags'] = SequenceLabelField(
                            coded_ner, sequence, "ner_tags")

                    # Add "tag label" to instance
                    if self.tag_label == 'ner':
                        instance_fields['tags'] = SequenceLabelField(
                            coded_ner, sequence)
                    elif self.tag_label == 'pos':
                        instance_fields['tags'] = SequenceLabelField(
                            pos_tags, sequence)
                    elif self.tag_label == 'chunk':
                        instance_fields['tags'] = SequenceLabelField(
                            coded_chunks, sequence)
                    if self.label_indexer is not None:
                        instance_fields["labels"] = self.label_indexer.index(
                            ner_tags, as_label_field=True)
                    yield Instance(instance_fields)

    @classmethod
    @overrides
    def from_params(cls, params: Params) -> 'WeakConll2003DatasetReader':
        token_indexers = TokenIndexer.dict_from_params(
            params.pop('token_indexers', {}))
        tag_label = params.pop('tag_label', None)
        feature_labels = params.pop('feature_labels', ())
        lazy = params.pop('lazy', False)
        coding_scheme = params.pop('coding_scheme', 'IOB1')
        label_indexer_params = params.pop('label_indexer', None)
        label_indexer = None
        if label_indexer_params is not None:
            label_indexer = LabelIndexer.from_params(label_indexer_params)
        convert_numbers = params.pop("convert_numbers", False)
        params.assert_empty(cls.__name__)
        return WeakConll2003DatasetReader(token_indexers=token_indexers,
                                          tag_label=tag_label,
                                          feature_labels=feature_labels,
                                          lazy=lazy,
                                          convert_numbers=convert_numbers,
                                          coding_scheme=coding_scheme,
                                          label_indexer=label_indexer)
