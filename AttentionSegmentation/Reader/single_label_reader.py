from __future__ import absolute_import
from typing import Optional, Dict
from overrides import overrides
import json
import io
import logging

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.common import Params, Tqdm
from allennlp.data.tokenizers import Token
from allennlp.data.fields import TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer,\
    TokenCharactersIndexer

logger = logging.getLogger(__name__)


class SingleLabelReader(DatasetReader):
    """Reads the data from a json file and
    converts them into a ``LabelField``

    Arguments:
        max_word_len (Optional[int]): The maximum word length allowed
        token_indexers (Dict[str, ``TokenIndexer``]): The token indexers
        lowercase (bool): Lowercase tokens or not (Default=True)
        tag_labels (str): The tags namespace
        label (str): The label we wish to predict for

    """

    def __init__(self,
                 label: str,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lowercase: bool = True,
                 label_namespace: str = "labels",
                 max_word_len: Optional[int] = None):
        super(SingleLabelReader, self).__init__()
        if max_word_len is None:
            logger.warning("It is strongly recommeded to set max_word_len")
            logger.warning("Not setting max_word_len can lead to extremely "
                           "long words, which are usually garbage")
        self._max_word_len = max_word_len
        self._lowercase = lowercase
        self._label = label
        self._label_namespace = label_namespace
        self._token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer(namespace="token_ids",
                                           lowercase_tokens=lowercase),
            "chars": TokenCharactersIndexer(namespace="token_chars")}

    @overrides
    def read(self, file_path, max_length=None):
        """Reads the data from a json file

        Converts the data into ``Instances`` of
        tokens and ``LabelField``

        Read the _read() function for more details

        """
        instances = self._read(file_path, max_length)
        if not isinstance(instances, list):
            instances = [instance for instance in Tqdm.tqdm(instances)]
        return instances

    @overrides
    def _read(self, file_path, max_length=None):
        """Loads data instances from a json file

        We expect the following structure for the json::

            sents (List[str]): The sentence
            entities (Dict[str, List[str]]): The extracted entites for each
                type
            labels (List[str]): The labels which exist for the example

        Arguments:
            file_path (str): The path to the data file
            max_length (Optional[int]): Max length beyond which
                the sentence will be truncated

        Returns
            List[``Instance``]: The instance list with "tokens", "entities"
            and "labels"

        """
        with io.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(fp=f, encoding="utf-8")
            instance_list = []
            for data_point in data:
                sentence = data_point["sents"]
                if max_length is not None:
                    sentence = sentence[:max_length]
                tokens = []
                for token in sentence:
                    if self._lowercase:
                        token = token.lower()
                        tokens.append(Token(token))
                text_field = TextField(tokens, self._token_indexers)
                # entities = data_point["entities"][self._label]
                indexed_label = 1 if self.label in data_point["labels"] \
                    else 0
                indexed_label_field = LabelField(
                    indexed_label, skip_indexing=True, label_namespace=self._)
                instance = Instance(
                    {
                        "tokens": text_field,
                        # "entities": entities,
                        "labels": indexed_label_field
                    }
                )
                instance_list.append(instance)
        return instance_list


