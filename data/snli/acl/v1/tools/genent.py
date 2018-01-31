#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse

import gzip
import json
import nltk
import copy

import numpy as np
import editdistance

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def negative_levenshtein(s1, s2):
    return - editdistance.eval(s1, s2)


def main(argv):
    def fmt(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('SNLI Entailment Candidate Generator', formatter_class=fmt)

    argparser.add_argument('--path', '-p', action='store', type=str, default='snli_1.0_train.jsonl.gz')
    argparser.add_argument('--seed', '-s', action='store', type=int, default=0)
    argparser.add_argument('--fraction', '-f', action='store', type=float, default=0.1)

    args = argparser.parse_args(argv)

    path = args.path
    seed = args.seed
    fraction = args.fraction

    obj_lst = []
    with gzip.open(path, 'rb') as f:
        for line in f:
            dl = line.decode('utf-8')
            obj = json.loads(dl)
            obj_lst += [obj]

    rs = np.random.RandomState(seed)

    nb_obj = len(obj_lst)
    # Round to the closest integer
    nb_samples = int(round(nb_obj * fraction))

    sampled_idxs = rs.choice(nb_obj, nb_samples, replace=False)
    sampled_obj_lst = [obj_lst[i] for i in sampled_idxs]

    res_obj_lst = []

    cache = dict()

    def stree_to_leaves(stree):
        if stree not in cache:
            tree = nltk.Tree.fromstring(stree)
            leaves = tree.leaves()
            cache[stree] = leaves
        return cache[stree]

    for s_obj in sampled_obj_lst:
        s1_parse, s2_parse = s_obj['sentence1_parse'], s_obj['sentence2_parse']
        # s1_parse_tokens, s2_parse_tokens = stree_to_leaves(s1_parse), stree_to_leaves(s2_parse)
        best_c, best_c_value = None, None

        for c_obj in obj_lst:
            c_s1_parse, c_s2_parse = c_obj['sentence1_parse'], c_obj['sentence2_parse']
            # c_s1_parse_tokens, c_s2_parse_tokens = stree_to_leaves(c_s1_parse), stree_to_leaves(c_s2_parse)

            # c_value = len(set(s2_parse_tokens).intersection(c_s1_parse_tokens))
            c_value = negative_levenshtein(s_obj['sentence2'], c_obj['sentence1'])

            if best_c is None or c_value > best_c_value:
                best_c = c_obj
                best_c_value = c_value

        res_obj = copy.deepcopy(s_obj)

        res_obj['annotator_labels'] = ['-']

        res_obj['sentence1'] = s_obj['sentence2']
        res_obj['sentence1_binary_parse'] = s_obj['sentence2_binary_parse']
        res_obj['sentence1_parse'] = s_obj['sentence2_parse']

        res_obj['sentence2'] = best_c['sentence1']
        res_obj['sentence2_binary_parse'] = best_c['sentence1_binary_parse']
        res_obj['sentence2_parse'] = best_c['sentence1_parse']

        res_obj_lst += [res_obj]

    for obj in res_obj_lst:
        print(json.dumps(obj), end='')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])