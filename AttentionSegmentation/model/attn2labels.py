import re


def get_binary_preds_from_attns(attns, tag, tol=0.01):
    tag_list = []
    for ix in range(len(attns)):
        if attns[ix] < tol:
            tag_list.append("O")
        else:
            if len(tag_list) > 0 and re.match(f".*-{tag}", tag_list[ix - 1]):
                tag_list.append(f"I-{tag}")
            else:
                tag_list.append(f"B-{tag}")
    return tag_list
