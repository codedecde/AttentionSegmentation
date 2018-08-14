from __future__ import absolute_import
import argparse
import sys
import io
from collections import OrderedDict
import re
import json

from AttentionSegmentation.Preprocess.constants import Constants


def get_arguments():
    def convert_to_boolean(args, name):
        if hasattr(args, name):
            assert getattr(args, name).lower() in set(["false", "true"]),\
                "Only boolean values allowed"
            val = True if getattr(args, name).lower() == "true" else False
            setattr(args, name, val)
        return args
    parser = argparse.ArgumentParser(description="Time Tagger")
    parser.add_argument('-src', '--src_file', action="store",
                        dest="src_file", type=str,
                        help="path to the source file", required=True)
    parser.add_argument('-tgt', '--tgt_file', action="store",
                        dest="tgt_file", type=str,
                        help="path to the target file", required=True)
    parser.add_argument('-ts', '--tag_set', action="store",
                        dest="tag_set", type=str,
                        default="ORG|LOC|MISC|PER",
                        help="Tag to store")
    args = parser.parse_args(sys.argv[1:])
    args.tag_set = set([x for x in args.tag_set.split("|")])
    return args


def get_data_from_file(src_file):
    data = []
    with io.open(src_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            sent = []
            for word_token in line.split(Constants.WORD_DELIMITER):
                word, tag = word_token.split(Constants.WORD_TAG_DELIMITER)
                sent.append((word, tag))
            data.append(sent)
    return data


def extract_labels(sent, label):
    extracted_labels = []
    current_label = None
    for w, t in sent:
        if re.match(f"B-{label}", t) is not None:
            if current_label is not None:
                extracted_labels.append(current_label)
                current_label = None
            current_label = w
        elif re.match(f"I-{label}", t) is not None:
            assert current_label is not None
            current_label += f" {w}"
        else:
            if current_label is not None:
                extracted_labels.append(current_label)
                current_label = None
    if current_label is not None:
        extracted_labels.append(current_label)
        current_label = None
    return extracted_labels


def convert_to_binary_data(data, label_set):
    converted_data = []
    for dp in data:
        label_entities = OrderedDict()
        for label in label_set:
            entities = extract_labels(dp, label)
            label_entities[label] = entities
        words = []
        binary_labels = set()
        label_golds = []
        for w, t in dp:
            words.append(w)
            for label in label_set:
                if re.match(f"[BI]-{label}$", t) is not None:
                    binary_labels.add(label)
        label_golds = list(binary_labels)
        converted_data.append(
            OrderedDict(
                {"sent": words,
                 "entities": label_entities,
                 "labels": label_golds}))
    return converted_data


def write_to_json_file(converted_data, target_file):
    with io.open(target_file, "w", encoding="utf-8", errors="ignore") as f:
        json.dump(obj=converted_data, fp=f, ensure_ascii=False, indent=4)


def main():
    """Convert the raw data into binary classification data

    Usage::

        ${PYTHONPATH} -m AttentionSegmentation.Preprocess.convert_to_binary
        --src_file ${SRC_FILE} --tgt_file ${TGT_FILE} --tag ${TAG}
    """
    args = get_arguments()
    raw_data = get_data_from_file(args.src_file)
    converted_data = convert_to_binary_data(raw_data, args.tag_set)
    write_to_json_file(converted_data, args.tgt_file)


if __name__ == "__main__":
    main()
