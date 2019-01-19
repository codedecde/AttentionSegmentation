from __future__ import absolute_import
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

from AttentionSegmentation.model.metrics import ConfusionMatrix


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


def _attn_to_rgb(attn_weights):
    attn_hex = str(hex(int(abs(attn_weights) * 255)))[2:]
    rgb = '#22aadd' + attn_hex
    return rgb


def _get_word_color(word, attn_weights):
    return '<span style="background-color:' + _attn_to_rgb(attn_weights) + \
        '">' + word + '</span>'


def colorize_text(text, attn_weights):
    """
    text: a string with the text to visualize
    attn_weights: a numpy vector in the range [0, 1]
        with one entry per word representing the attention weight
    """
    words = text.split()
    assert len(words) == len(attn_weights)
    html_blocks = [''] * len(words)
    for i in range(len(words)):
        html_blocks[i] += _get_word_color(words[i], attn_weights[i])
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


if __name__ == "__main__":
    # colorized_text_to_webpage(
    #     'This is a test', [0.1, 0.2, 0.1, 0.7], vis_page="WebOuts/test.html")
    lst_txt = [
        "This is a test", "And another one", "And one last for good luck"
    ]
    attn_weights = [[1, 1, 1, 10], [4, 2, 1], [3, 2, 1, 4, 5, 9]]
    for ix in range(len(attn_weights)):
        dr = float(sum(attn_weights[ix]))
        for jx in range(len(attn_weights[ix])):
            attn_weights[ix][jx] /= dr
    colorized_list_to_webpage(
        lst_txt, attn_weights, vis_page="WebOuts/test.html"
    )

    # colorized_text_to_webpage(
    #     'This is a test', [0.1, 0.2, 0.1, 0.7], vis_page="WebOuts/test.html")
