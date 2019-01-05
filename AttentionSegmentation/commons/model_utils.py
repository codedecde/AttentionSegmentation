from __future__ import absolute_import
import io
import os
import re
import torch
import logging
from typing import Any, Dict, Union, List
from allennlp.data import Vocabulary
from allennlp.common.util import namespace_match

NAMESPACE_PADDING_FILE = 'non_padded_namespaces.txt'
logger = logging.getLogger(__name__)


def set_vocab_from_filename(vocab: Vocabulary, namespace_filename: str,
                            load_dir: str, non_padded_namespaces: str):
    """Set up the vocabulary from a file

    Arguments:
        vocab: The vocabulary
        namespace_filename: The file containing all the namespaces
            to be loaded
        load_dir: The directory to load the vocab from
        non_padded_namespaces: The namespaces that are not padded
            (like labels etc)
    Returns:
        ``Vocabulary``: The loaded vocabulary
    """
    namespace = namespace_filename.replace('.txt', '')
    if any(namespace_match(pattern, namespace)
           for pattern in non_padded_namespaces):
        is_padded = False
    else:
        is_padded = True
    filename = os.path.join(load_dir, namespace_filename)
    vocab.set_from_file(filename, is_padded, namespace=namespace)
    return vocab


def construct_vocab(src_dir: str, tgt_dir: str):
    """Construct vocab, taking in src_directory, and tgt_directory

    Loads the vocab from the src_dir, if the namespace is present,
    else loads from the tgt_dir

    Arguments:
        src_dir (str): The source directory.
        tgt_dir (str): The target directory.
    Returns:
        ``Vocabulary``: The loaded vocabulary
    """
    src_files = set()
    for file in os.listdir(src_dir):
        if file != NAMESPACE_PADDING_FILE:
            src_files.add(file)
    # load the non_padded_namespaces.txt
    read_params = {"encoding": "utf-8", "errors": "ignore"}
    tgt_file = os.path.join(tgt_dir, NAMESPACE_PADDING_FILE)
    with io.open(tgt_file, 'r', **read_params) as namespace_file:
        non_padded_namespaces = [namespace_str.strip()
                                 for namespace_str in namespace_file]
    vocab = Vocabulary(non_padded_namespaces=non_padded_namespaces)
    for namespace_filename in os.listdir(tgt_dir):
        if namespace_filename == NAMESPACE_PADDING_FILE:
            continue
        load_dir = src_dir if namespace_filename in src_files else tgt_dir
        vocab = set_vocab_from_filename(
            vocab, namespace_filename, load_dir, non_padded_namespaces)
    return vocab


def load_model_from_existing(src_model: Union[str, Dict[str, Any]],
                             tgt_model: torch.nn.Module,
                             layers: List[str] = None) -> torch.nn.Module:
    """Loads layers specified by ``layers`` from the source_model to the tgt_model.

        Arguments:
            src_model (Union[str, Dict[str, Any]]): The source state dict,
                or filename to the state dict
            tgt_model (``torch.nn.Module``):  The nn module to which
                we load the dict
            layers (List[str]): The (prefix of) layers to load
        Returns:
            ``torch.nn.Module``: The output nn module, with weights loaded.
    """
    if isinstance(src_model, str):
        src_model = torch.load(
            src_model, map_location=lambda storage, loc: storage)
    common_layers = list(
        set(src_model.keys()) & set(tgt_model.state_dict().keys()))
    if layers is not None:
        _common_layers = []
        for layer in common_layers:
            if any(re.match(f"^{x}*", layer) for x in layers):
                _common_layers.append(layer)
        common_layers = _common_layers
    intersection = {}
    for layer in sorted(common_layers):
        logger.info(f"Loading layer: {layer}")
        intersection[layer] = src_model[layer]
    tgt_model.load_state_dict(intersection, strict=False)
    return tgt_model
