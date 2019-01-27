from __future__ import absolute_import
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import pdb
import json
import re
import argparse
import sys

from allennlp.data.iterators import BasicIterator
from allennlp.data.instance import Instance
from allennlp.models.model import Model
from allennlp.common.tqdm import Tqdm


from AttentionSegmentation.model.metrics import ConfusionMatrix

colors2rgb = {}

colors2rgb['purple'] = '#712f79'  # Pred tag, gold no tag
colors2rgb['brickRed'] = '#d2405f'  # Pred no tag, gold tag
colors2rgb['yellowGreen'] = '#e0ff4f'  # Both tag


def colorized_predictions_to_webpage(
        predictions, vis_page="visualize.html"):
    """This generates the visualization web page from predictions

    Arguments:
        predictions (List[Dict[str, Any]]): A list of predictions.
            Each prediction contains:
                * text (List[str]): list of tokens
                * pred (str): The predicted token
                * gold (str): The gold token
                * attn (List[float]): The list of float tokens
                * pred_labels (List[str]) : The list of predicted
                    labels
                * gold_labels (List[str]) : The list of gold labels
        vis_page (str): The final output page

    """
    with open(vis_page, "w") as f:
        purple = colors2rgb['purple']
        brickRed = colors2rgb['brickRed']
        yellowGreen = colors2rgb['yellowGreen']
        header = (
            '<html>\n'
            '<head>\n'
            '<style>\n'  # The CSS element
            '   correct { '
            '       color: #8dde28; '
            '       padding-right: 5px; '
            '       padding-left: 5px '
            '   }\n'
            '   incorrect { '
            '       color: #e93f3f; '
            '       padding-right: 5px; '
            '       padding-left: 5px '
            '   }\n'
            '   body { color: color:#000000}\n'
            '   .tooltip { '
            '       position: relative; '
            '       display: inline-block; '
            # '       border-bottom: 1px dotted black;'
            '   }\n'
            '   .tooltip .tooltiptext {  '
            '       visibility: hidden;  '
            '       width: 120px;  '
            '       background-color: black; '
            '       color: #fff; '
            '       text-align: center;  '
            '       border-radius: 6px;  '
            '       padding: 5px 0;  '
            '       position: absolute;  '
            '       z-index: 1;  '
            '       top: 150%; '
            '       left: 50%; '
            '       margin-left: -60px;  '
            '   }\n'
            '   .tooltip .tooltiptext::after { '
            '       content: " ";    '
            '       position: absolute;  '
            '       bottom: 100%;  /* At the top of the tooltip */   '
            '       left: 50%;   '
            '       margin-left: -5px;   '
            '       border-width: 5px;   '
            '       border-style: solid; '
            '       border-color: transparent transparent black transparent; '
            '   }\n'
            '   .tooltip:hover .tooltiptext {  '
            '       visibility: visible; '
            '   }\n'
            '</style>\n'
            '</head>\n'
            '<body>'
            'Key:</br>'
            f'<span style="background-color:{purple}; padding-left: 10px; padding-right: 10px; color:white" >Pred tag, Gold no tag</span> </br>'
            f'<span style="background-color:{brickRed}; padding-left: 10px; padding-right: 10px; color:white" >Pred no tag, Gold tag</span> </br>'
            f'<span style="background-color:{yellowGreen}; padding-left: 10px; padding-right: 10px; color:black">Both Correct tag</span> </br>'
            '</br>'
        )
        f.write(header)
        for pred in predictions:
            txt = " ".join(pred["text"])
            attn_weights = pred["attn"]
            pred_label = pred["pred"]
            gold_label = pred["gold"]
            pred_tags = pred["pred_labels"]
            gold_tags = pred["gold_labels"]
            html = colorize_text(txt, attn_weights, pred_tags, gold_tags)
            if pred_label == gold_label:
                pred_gold = (
                    '<correct>'
                    f' {pred_label} '
                    f' {gold_label} '
                    '</correct>'
                )
            else:
                pred_gold = (
                    '<incorrect>'
                    f' {pred_label} '
                    f' {gold_label} '
                    '</incorrect>'
                )
            f.write(f"{html}{pred_gold}<br>")
        footer = "</body></html>"
        f.write(footer)


class html_visualizer(object):
    """This collects the different visualization methods for easy visualization
    """

    def __init__(self, vocab, reader):
        self._vocab = vocab
        self._iterator = BasicIterator(batch_size=32)
        self._iterator.index_with(self._vocab)
        self._reader = reader

    def _get_text_from_instance(self, instance: Instance) -> List[str]:
        """Helper function to extract text from an instance
        """
        return list(map(lambda x: x.text, instance.fields['tokens'].tokens))

    def visualize_data(
        self,
        instances: List[Instance],
        model: Model,
        filename: str,
        cuda_device: int = -1
    ):
        """This function helps visualize the attention maps
        We use a basic itereator, since a bucket iterator shuffles
        data, even for shuffle=False

        Arguments:
            data (List[Instance]) : The list of instances for inference
            filename (str) : The html file to output to
            cuda_device (int) : The GPU being used
        """
        iterator = self._iterator(
            instances,
            num_epochs=1,
            shuffle=False,
            cuda_device=cuda_device,
            for_training=False
        )
        model.eval()
        num_batches = self._iterator.get_num_batches(instances)
        inference_generator_tqdm = Tqdm.tqdm(iterator, total=num_batches)
        predictions = []
        index = 0
        index_labeler = self._reader.get_label_indexer()
        correct_counts = 0.
        for batch in inference_generator_tqdm:
            # Currently I don't support multi-gpu data parallel
            output_dict = model.decode(model(**batch))
            for ix in range(len(output_dict["preds"])):
                text = self._get_text_from_instance(instances[index])
                label_num = instances[index].fields['labels'].labels[0]
                # FIXME: Currently supporting binary classification
                assert len(instances[index].fields['labels'].labels) == 1
                index += 1
                pred = output_dict["preds"][ix]
                attn = output_dict["attentions"][ix]
                gold = "O"
                if label_num < len(index_labeler.ix2tags):
                    gold = index_labeler.ix2tags[label_num]
                if pred == gold:
                    correct_counts += 1.
                prediction = {
                    "text": text,
                    "pred": pred,
                    "attn": attn,
                    "gold": gold
                }
                predictions.append(prediction)
        if filename != "":
            colorized_predictions_to_webpage(
                predictions, vis_page=filename)


def plot_data_point(data_point, mode="hierarchical", **kwargs):
    if mode == "hierarchical":
        plot_hierarchical_attn(
            **data_point, fontsize=12, **kwargs)
    else:
        raise NotImplementedError("Work in progress")


def plot_hierarchical_attn(
        sentences: List[str], sent_attn: List[float],
        word_attns: List[List[float]], preds: List[str], golds: List[str],
        sent_thresh: int = 15, max_sent_len: int = 10,
        filename: Optional[str] = None,
        label_probs: Optional[List[float]] = None, **kw):
    """Plots the attention representation, given arguments

    Arguments:
        sentences (List[str]): The sentences
        sent_attn (List[float]): The Sentence level attn
        word_attns (List[List[float]]): The word level
            attention
        preds (List[str]): The predictions
        golds (List[str]): The gold labels
        sent_thresh (int): The number of sentences to print
        max_sent_len (int): Max length of sentence to print
        filename (Optional[str]): Save figure to this file
        label_probs (Optional[List[float]]): Unused. Required for
            consistency

    """
    plt.close()
    num_sentences = len(sentences)
    truncated_word_attns = []
    truncated_sents = []
    for sent, attns in zip(sentences, word_attns):
        top_attn_indices = np.argsort(np.array(attns))[::-1][:sent_thresh - 1]
        top_index = max(top_attn_indices)
        truncated_sent = sent[
            max(0, top_index - sent_thresh + 1): top_index + 1]
        truncated_sents.append(truncated_sent)
        truncated_attn = attns[
            max(0, top_index - sent_thresh + 1): top_index + 1]
        truncated_word_attns.append(truncated_attn)
        max_sent_len = max(max_sent_len, len(truncated_sent))
    attn_matrix = np.zeros((num_sentences, max_sent_len + 1))
    attn_matrix[:, 0] = np.array(sent_attn[::-1])
    for ix in range(0, len(sentences)):
        iterator_end = min(
            len(truncated_word_attns[ix]), attn_matrix.shape[-1] - 1)
        for jx in range(0, iterator_end):
            attn_matrix[num_sentences - ix - 1, jx + 1] = \
                truncated_word_attns[ix][jx]
    plt.figure(figsize=(24, 12))
    title_string = "Attention Map:"
    gold_string = " , ".join(golds)
    pred_string = " , ".join(preds)
    title_string = (
        f"{title_string}        Predictions: {pred_string}"
        f"        Gold Labels: {gold_string}"
    )
    plt.title(title_string)
    c = plt.pcolor(
        attn_matrix, edgecolors='k',
        linewidths=4, cmap='Blues', vmin=0.0, vmax=1.0)
    c.update_scalarmappable()
    ax = c.axes
    word_thresh = 6
    fmt = f"%{word_thresh}s"
    index = 0
    reversed_sentences = truncated_sents[::-1]
    for p, color, value in zip(
            c.get_paths(), c.get_facecolors(), c.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        r, c = index // (max_sent_len + 1), index % (max_sent_len + 1)
        if c > 0:
            if np.all(color[:3] > 0.5):
                color = (0.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)
            if c - 1 < len(reversed_sentences[r]):
                text = reversed_sentences[r][c - 1][:word_thresh]
                ax.text(
                    x, y, fmt % text, ha="center", va="center",
                    color=color, **kw)
        index += 1
    if filename is not None:
        plt.savefig(filename)


def plot_confusion_matrix(confusion_matrix: ConfusionMatrix,
                          filename: Optional[str] = None):
    plt.close()

    gold_labels = confusion_matrix.labels
    pred_labels = confusion_matrix.labels
    df_cm = pd.DataFrame(
        confusion_matrix.confusion_matrix, index=[i for i in gold_labels],
        columns=[i for i in pred_labels]
    )
    plt.figure(figsize=(10, 7))
    ax = sn.heatmap(df_cm, annot=True, fmt="3.1f")
    ax.set(xlabel="Pred Labels", ylabel="Gold Labels")
    plt.title("Confusion Matrix for Classification")
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    return confusion_matrix


def _attn_to_rgb(attn_weights, pred_tag, gold_tag):
    pred_tag = re.sub(".*-", "", pred_tag)
    gold_tag = re.sub(".*-", "", gold_tag)
    attn_hex = str(hex(int(abs(attn_weights) * 255)))[2:]
    if pred_tag == gold_tag:
        if pred_tag != "O":
            rgb = colors2rgb['yellowGreen']  # + attn_hex
        else:
            rgb = '#22aadd' + attn_hex
    else:
        if pred_tag == "O":
            rgb = colors2rgb["brickRed"]  # + attn_hex
        elif gold_tag == "O":
            rgb = colors2rgb["purple"]  # + attn_hex
        else:
            pdb.set_trace()
    return rgb


def _get_word_color(word, attn_weights, pred_tag, gold_tag):
    color = _attn_to_rgb(attn_weights, pred_tag, gold_tag)
    return (
        '<div class="tooltip">'
        f'    <span style="background-color:{color}">{word}</span>'
        f'    <span class="tooltiptext">{attn_weights:2.2f}</span>'
        f'</div>'
    )


def colorize_text(text, attn_weights, pred_tags, gold_tags):
    """
    text: a string with the text to visualize
    attn_weights: a numpy vector in the range [0, 1]
        with one entry per word representing the attention weight
    """
    words = text.split()
    assert len(words) == len(attn_weights)
    html_blocks = [''] * len(words)
    for i in range(len(words)):
        html_blocks[i] += _get_word_color(
            words[i], attn_weights[i], pred_tags[i], gold_tags[i]
        )
    return ' '.join(html_blocks)


def get_colorized_text_as_html(text, attn_weights):
    return '<html><body style="color:#000000">' + \
        colorize_text(text, attn_weights) + '</body></html>'


def colorized_text_to_webpage(text, attn_weights, vis_page='visualize.html'):
    """
    # Sample code:
    from visualize_attns import  colorized_text_to_webpage
    colorized_text_to_webpage('This is a test', [0.1, 0.2, 0.1, 0.7])
    """
    with open(vis_page, 'w') as f:
        f.write(get_colorized_text_as_html(text, attn_weights))


def colorized_list_to_webpage(
        lst_txt, lst_attn_weights, vis_page="visualize.html"):
    """This generates the output of a list of sentences as a web page
    """
    with open(vis_page, 'w') as f:
        for txt, attn_weights in zip(lst_txt, lst_attn_weights):
            html = get_colorized_text_as_html(txt, attn_weights)
            f.write(f"{html}<br>")


def get_arguments():
    parser = argparse.ArgumentParser(description="Time Tagger")
    parser.add_argument('-src', '--src', action="store",
                        dest="src", type=str,
                        help="path to the source predictions", required=True)
    parser.add_argument('-tgt', '--tgt', action="store",
                        dest="tgt", type=str,
                        help="path to the target predictions", required=True)
    args = parser.parse_args(sys.argv[1:])
    return args


if __name__ == "__main__":
    args = get_arguments()
    fil = args.src
    with open(fil, 'r') as f:
        predictions = json.load(f)
    fil = args.tgt
    colorized_predictions_to_webpage(
        predictions,
        args.tgt
    )
