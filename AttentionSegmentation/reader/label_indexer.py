from __future__ import absolute_import
from typing import List, Union
import re
import pdb

from allennlp.data.fields import MultiLabelField
from allennlp.common import Params


class LabelIndexer(object):
    """
    This class indexes labels for multi label classification
    """

    def __init__(
        self,
        label_namespace: str,
        tags: List[str]
    ):
        self.tags = sorted(tags)
        self.tags2ix = {w: ix for ix, w in enumerate(self.tags)}
        self.ix2tags = {self.tags2ix[t]: t for t in self.tags2ix}
        self.label_namespace = label_namespace

    def get_num_tags(self) -> int:
        # self.tags2ix does not contain the 'O' tag
        return len(self.tags2ix) + 1

    def index(
        self,
        ner_tags: List[str],
        as_label_field: bool
    ) -> Union[List[int], MultiLabelField]:
        """
        Takes in a list of tags ([B-PER, I-PER, O, O, B-LOC, I-LOC]),
        performs a regex match against the ner tags (.*-TAG), and
        generates the label accordingly

        Parameters:
            ner_tags (List[str]): The list of NER Tags
            as_label_field (bool): If True, returns a MultiLabelField,
                otherwise returns a list of tag indices

        Returns:
            indices (Union[List[int], MultiLabelField]): Returns
                either a list of indexed labels, or a MultiLabelField
                instance

        """
        indices = set()
        for gold_tag in ner_tags:
            for tag in self.tags2ix:
                if re.match(f".*-{tag}", gold_tag) is not None:
                    indices.add(self.tags2ix[tag])
        if len(indices) > 0:
            indices = list(indices)
        else:
            indices = [len(self.tags2ix)]
        if as_label_field:
            indices = MultiLabelField(
                labels=indices,
                label_namespace=self.label_namespace,
                skip_indexing=True,
                num_labels=self.get_num_tags())
        return indices

    @classmethod
    def from_params(cls, params: Params) -> 'LabelIndexer':
        label_namespace = params.pop("label_namespace")
        tags = params.pop("tags")
        params.assert_empty(cls.__name__)
        return cls(label_namespace=label_namespace, tags=tags)
