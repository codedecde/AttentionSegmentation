from __future__ import absolute_import
from AttentionSegmentation.model.inference import BasicAttentionModelRunner
import argparse
import sys
import json


def get_arguments():
    parser = argparse.ArgumentParser(description="Time Tagger")
    parser.add_argument('-bd', '--base_dir', action="store",
                        dest="base_dir", type=str,
                        help="path to the experiment directory", required=True)
    parser.add_argument('-vf', '--val_file', action="store", dest="val_file",
                        type=str, help="The test file", required=True)
    parser.add_argument('-hf', '--html_file', action="store", dest="html_file",
                        type=str, default="",
                        help="The html file used for output")
    parser.add_argument("-pf", "--pred_file", action="store", dest="pred_file",
                        type=str, default="",
                        help="The prediction json file")
    args = parser.parse_args(sys.argv[1:])
    return args


if __name__ == "__main__":
    args = get_arguments()
    base_dir = args.base_dir
    runner = BasicAttentionModelRunner.load_from_dir(base_dir)
    valid_file = args.val_file
    output_file = args.html_file
    # "./WebOuts/visualize.html"
    predictions = runner._process_file(valid_file, output_file)
    if args.pred_file != "":
        with open(args.pred_file, "w") as f:
            json.dump(predictions, f, indent=4, ensure_ascii=True)
