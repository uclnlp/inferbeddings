#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging

import numpy as np

from sklearn.cross_validation import StratifiedKFold, train_test_split


def read_triples(path):
    with open(path, 'rt') as f:
        lines = f.readlines()
    triples = [(s.strip(), p.strip(), o.strip()) for [s, p, o] in [l.split() for l in lines]]
    return triples


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Stratified K-Folder for Knowledge Graphs', formatter_class=formatter)
    argparser.add_argument('triples', action='store', type=str, default=None)

    args = argparser.parse_args(argv)

    triples_path = args.triples

    triples = read_triples(triples_path)
    triples_set = set(triples)

    entities = {s for (s, p, o) in triples} | {o for (s, p, o) in triples}
    predicates = {p for (s, p, o) in triples}

    all_triples, all_labels = [], []
    for s in entities:
        for p in predicates:
            for o in entities:
                all_triples += [(s, p, o)]
                label = 1 if (s, p, o) in triples_set else 0
                all_labels += [label]

    all_labels_np = np.array(all_labels)
    skf = StratifiedKFold(n_folds=10, y=all_labels_np, shuffle=True, random_state=0)

    all_triples_np = np.array(all_triples)
    for fold_no, (train_valid_idx, test_idx) in enumerate(skf):
        train_idx, valid_idx, _, _ = train_test_split(train_valid_idx, np.ones(train_valid_idx.shape[0]),
                                                      test_size=test_idx.shape[0], random_state=0)

        train_triples = all_triples_np[train_idx]
        train_labels = all_labels_np[train_idx]

        valid_triples = all_triples_np[valid_idx]
        valid_labels = all_labels_np[valid_idx]

        test_triples = all_triples_np[test_idx]
        test_labels = all_labels_np[test_idx]

        train_lines = ['{}\t{}\t{}\t{}'.format(s, p, o, l) for [s, p, o], l in zip(train_triples, train_labels)]
        valid_lines = ['{}\t{}\t{}\t{}'.format(s, p, o, l) for [s, p, o], l in zip(valid_triples, valid_labels)]
        test_lines = ['{}\t{}\t{}\t{}'.format(s, p, o, l) for [s, p, o], l in zip(test_triples, test_labels)]

        if not os.path.exists('stratified_folds/{}'.format(str(fold_no))):
            os.mkdir('stratified_folds/{}'.format(str(fold_no)))

        with open('stratified_folds/{}/umls_train.tsv'.format(str(fold_no)), 'w') as f:
            f.writelines(['{}\n'.format(line) for line in train_lines])

        with open('stratified_folds/{}/umls_valid.tsv'.format(str(fold_no)), 'w') as f:
            f.writelines(['{}\n'.format(line) for line in valid_lines])

        with open('stratified_folds/{}/umls_test.tsv'.format(str(fold_no)), 'w') as f:
            f.writelines(['{}\n'.format(line) for line in test_lines])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
