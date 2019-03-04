from __future__ import absolute_import
from typing import List, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
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


# from AttentionSegmentation.model.attn2labels \
#     import get_binary_preds_from_attns

colors2rgb = {}

colors2rgb['purple'] = '#712f79'  # Pred tag, gold no tag
colors2rgb['brickRed'] = '#d2405f'  # Pred no tag, gold tag
colors2rgb['yellowGreen'] = '#e0ff4f'  # Both tag

tag2color = OrderedDict({
    "PER": "#e88a1a",
    "ORG": "#005542",
    "LOC": "#10316b",
    "MISC": "#f6b8d1"
})
dark_backgrounds = set(["ORG", "LOC"])


def strip(string):
    return re.sub(".*-", "", string)


def get_html_from_pred(pred, debug=True):
    writebuf = []
    predicted_tags, predicted_probs = [list(x) for x in zip(*pred["pred"])]
    for ix in range(len(pred["text"])):
        pred_label = strip(pred["pred_labels"][ix])
        gold_label = strip(pred["gold_labels"][ix])
        correct = False
        if pred_label == gold_label:
            correct = True
        word = pred["text"][ix]
        html = ['<div class="tooltip">']
        attn_at_point = [(pred["attn"][t][ix], t) for t in pred["attn"]
                         if t in predicted_tags]
        if len(attn_at_point) == 0:
            attn_weight, tag = max([(pred["attn"][t][ix], t) for t in pred["attn"]])
            attn_hex = str(hex(int(abs(attn_weight) * 255)))[2:]
            # we use a neutral gray color for this case
            attn_color = "#3c415e" + attn_hex
        else:
            attn_weight, tag = max(attn_at_point)
            attn_hex = str(hex(int(abs(attn_weight) * 255)))[2:]
            attn_color = tag2color[tag] + attn_hex
        html.append('<span style="padding:2px">')
        if correct and pred_label != "O" and debug:
            html.append('<underline style=text-decoration-color:#8dde28>')
        else:
            if pred_label != "O" and debug:
                color = tag2color[pred_label]
                html.append(f'<overline style="text-decoration-color:{color}">')
            if gold_label != "O" and debug:
                color = tag2color[gold_label]
                html.append(f'<underline style="text-decoration-color:{color}">')
        html.append(f"<span style=background-color:{attn_color}>")
        html.append(word)
        html.append("</span>")
        if correct and pred != "O" and debug:
            html.append('</underline>')
        else:
            if gold_label != "O" and debug:
                html.append('</underline>')
            if pred_label != "O" and debug:
                html.append("</overline>")
        html.append('</span>')
        html.append('<span class="tooltiptext">')
        for tag in pred["attn"]:
            attnval = "{0:2.2f}".format(pred["attn"][tag][ix])
            string = "{0:>4s}: {1:4s}".format(tag, attnval)
            html.append(f"{string} <br>")
        html.append('</span>')
        html.append('</div>')
        writebuf.append("".join(html))
    writebuf.append(" ")
    writebuf.append("[")

    for t in predicted_tags:
        if t in pred["gold"]:
            writebuf.append(f"<correct>{t} </correct>")
        else:
            writebuf.append(f"<incorrect>{t} </incorrect>")
    writebuf.append("]")
    writebuf.append(" ")
    writebuf.append("[")
    for t in pred["gold"]:
        if t in predicted_tags:
            writebuf.append(f"<correct>{t} </correct>")
        else:
            writebuf.append(f"<incorrect>{t} </incorrect>")
    writebuf.append("]")
    return "".join(writebuf)


def colorized_predictions_to_webpage(
        predictions, webpage="visualize.html", debug=True):
    header = (
        '<html>\n'
        '<head>\n'
        '<style>\n'
        ' correct { \n'
        '     color: #8dde28; \n'
        '     padding-right: 5px; \n'
        '     padding-left: 5px \n'
        ' }\n'
        ' incorrect { \n'
        '     color: #cf3030; \n'
        '     padding-right: 5px; \n'
        '     padding-left: 5px \n'
        ' }\n'
        ' overline {\n'
        '    text-decoration: overline;\n'
        ' }\n'
        ' underline {\n'
        '    text-decoration: underline;\n'
        ' }\n'
        ' body { color: color:#000000}\n'
        ' .tooltip { \n'
        '     position: relative; \n'
        '     display: inline-block; \n'
        ' }\n'
        ' .tooltip .tooltiptext {  \n'
        '     visibility: hidden;  \n'
        '     width: 120px;  \n'
        '     background-color: black; \n'
        '     color: #fff; \n'
        '     text-align: center;  \n'
        '     border-radius: 6px;  \n'
        '     padding: 5px 0;  \n'
        '     position: absolute;  \n'
        '     z-index: 1;  \n'
        '     top: 150%; \n'
        '     left: 50%; \n'
        '     margin-left: -60px;  \n'
        ' }\n'
        ' .tooltip .tooltiptext::after { \n'
        '     content: " ";    \n'
        '     position: absolute;  \n'
        '     bottom: 100%;\n'
        '     left: 50%;   \n'
        '     margin-left: -5px;   \n'
        '     border-width: 5px;   \n'
        '     border-style: solid; \n'
        '     border-color: transparent transparent black transparent; \n'
        ' }\n'
        ' .tooltip:hover .tooltiptext {  \n'
        '     visibility: visible; \n'
        ' }\n'
        '</style>\n'
        '</head>\n'
    )
    body = ["<body>"]
    for tag in tag2color:
        color = tag2color[tag]
        if tag in dark_backgrounds:
            text_background = "white"
        else:
            text_background = "black"
        text = f'<span style="background-color:{color}; color: {text_background}">{tag}</span> '
        body.append(text)
    body.append("<br><br>")
    for pred in predictions:
        html = get_html_from_pred(pred, debug=debug)
        body.append(f"{html}<br><br>")
    footer = ["</body></html>"]
    with open(webpage, "w") as f:
        f.write("\n".join([header] + body + footer))


def colorized_predictions_to_webpage_binary(
        predictions, vis_page="visualize.html"):
    """This generates the visualization web page from predictions

    Arguments:
        predictions (List[Dict[str, Any]]): A list of predictions.
            Each prediction contains:
                * text (List[str]): list of tokens
                * pred (List[str]): The predicted tokens
                * gold (List[str]): The gold tokens
                * attn (Dict[str, List[float]]): The attentions,
                    by tags
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
            '<span'
            f'  style="background-color:{purple};'
            '   padding-left: 10px;'
            '   padding-right: 10px;'
            '   color:white" >Pred tag, Gold no tag</span></br>'
            '<span'
            f'   style="background-color:{brickRed};'
            '    padding-left: 10px;'
            '    padding-right: 10px;'
            '    color:white" >Pred no tag, Gold tag</span> </br>'
            '<span'
            f'   style="background-color:{yellowGreen};'
            '    padding-left: 10px;'
            '    padding-right: 10px;'
            '    color:black">Both Correct tag</span> </br>'
            '</br>'
        )
        f.write(header)
        for pred in predictions:
            txt = " ".join(pred["text"])
            attn_weights = list(pred["attn"].values())[0]
            pred_label = pred["pred"][0]
            gold_label = pred["gold"][0]
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
