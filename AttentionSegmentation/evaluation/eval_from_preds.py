from __future__ import absolute_import
import json
import argparse
import sys

from AttentionSegmentation.evaluation.conlleval_perl \
    import countChunks, evaluate
from AttentionSegmentation.model.attn2labels import \
    get_binary_preds_from_attns
from AttentionSegmentation.visualization.visualize_attns \
    import colorized_predictions_to_webpage


def get_arguments():
    parser = argparse.ArgumentParser(description="Time Tagger")
    parser.add_argument('-s', '--src', action="store",
                        dest="src", type=str, required=True,
                        help="src file"
                        )
    parser.add_argument('-t', '--tag', action="store",
                        dest="tag", type=str, required=True,
                        help="The tag to test"
                        )
    parser.add_argument('-tol', '--tol', action="store",
                        dest="tol", type=float, default=0.01,
                        help="The attention threshold"
                        )
    parser.add_argument('-tgt', '--tgt', action="store",
                        dest="tgt", type=str,
                        help="path to the target predictions", required=True)
    args = parser.parse_args(sys.argv[1:])
    return args


if __name__ == "__main__":
    args = get_arguments()
    with open(args.src, "r") as f:
        preds = json.load(f)
    # with open("output.txt", "w") as f:
    buf = []
    boundary = "-X-"
    for pred in preds:
        if pred["pred"] == "O":
            pred["pred_labels"] = ["O"] * len(pred["attn"])
        else:
            pred["pred_labels"] = get_binary_preds_from_attns(
                pred["attn"], args.tag, args.tol
            )
        tmp = [
            [txt, gold, pred] for txt, pred, gold in zip(
                pred["text"], pred["gold_labels"], pred["pred_labels"]
            )
        ]
        buf += tmp + [[boundary, "O", "O"]]
        # buf.append("\n".join(tmp))
    # buf = "\n\n".join(buf).split("\n")
    correctChunk, foundGuessed, foundCorrect, correctTags, tokenCounter = \
        countChunks(buf)
    evaluate(
        correctChunk, foundGuessed,
        foundCorrect, correctTags, tokenCounter,
        latex=False
    )
    colorized_predictions_to_webpage(
        preds,
        args.tgt
    )
        # f.write("\n\n".join(buf))
