#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import string
import sys
import unicodedata

from word2number import w2n

# def is_list_in_string(text, candidate):
#     return any([all(xx in text for xx in x.split(" ")) if isinstance(x, str) else all([xx in text for xx in x]) for x in candidate])


def is_string_in_string(text, candidate):
    return all(x in text for x in candidate.split(" "))


def is_list_in_string(text, candidate):
    return any(
        [
            is_string_in_string(text, x) if isinstance(x, str) else is_list_in_string(text, x)
            for x in candidate
        ]
    )


def clean_punctuation(value):
    punctuation = string.punctuation
    punctuation = punctuation.replace("'", "")
    value = re.sub(f"[{punctuation}]", " ", value)
    return value


if __name__ == "__main__":

    pred_gt_json_file = sys.argv[1]

    with open(pred_gt_json_file, "r") as f:
        pred_gt = json.load(f)

    acc = 0
    for line in pred_gt:

        pred = line[0]
        gt = line[1]

        # pred = clean_punctuation(pred)
        pred = pred.lower()

        if isinstance(gt, list):
            pass
        else:
            gt = [
                gt,
            ]
        gt = [clean_punctuation(x) for x in gt]
        gt = [x.lower().strip() for x in gt]

        try:
            gt_number = [str(w2n.word_to_num(x.lower())) for x in gt]
        except:
            gt_number = gt
            pass

        if is_list_in_string(pred, gt):
            acc += 1
        elif is_list_in_string(pred, gt_number):
            acc += 1
        else:
            print("======================================================")
            print(f"{line[0]=}")
            print(f"{line[1]=}")

    print("======================================================")
    print(f"{acc=}")
    print(f"{len(pred_gt)=}")
    print("======================================================")

    acc = acc / len(pred_gt) * 100

    print("======================================================")
    print(f"{acc=}")
    print("======================================================")
