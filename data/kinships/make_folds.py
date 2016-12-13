#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging

import numpy as np

from sklearn.cross_validation import KFold, train_test_split


def read_triples(path):
    with open(path, 'rt') as f:
        lines = f.readlines()
    triples = [(s.strip(), p.strip(), o.strip()) for [s, p, o] in [l.split() for l in lines]]
    return triples


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('K-Folder for Knowledge Graphs', formatter_class=formatter)
    argparser.add_argument('triples', action='store', type=str, default=None)

    args = argparser.parse_args(argv)

    triples_path = args.triples

    triples = read_triples(triples_path)
    nb_triples = len(triples)

    kf = KFold(n=nb_triples, n_folds=10, random_state=0, shuffle=True)

    triples_np = np.array(triples)

    for fold_no, (train_idx, test_idx) in enumerate(kf):
        train_valid_triples = triples_np[train_idx]
        test_triples = triples_np[test_idx]

        train_triples, valid_triples, _, _ = train_test_split(train_valid_triples,
                                                              np.ones(train_valid_triples.shape[0]),
                                                              test_size=len(test_triples), random_state=0)

        train_lines = ['{}\t{}\t{}'.format(s, p, o) for [s, p, o] in train_triples]
        valid_lines = ['{}\t{}\t{}'.format(s, p, o) for [s, p, o] in valid_triples]
        test_lines = ['{}\t{}\t{}'.format(s, p, o) for [s, p, o] in test_triples]

        if not os.path.exists('folds/{}'.format(str(fold_no))):
            os.mkdir('folds/{}'.format(str(fold_no)))

        with open('folds/{}/nations_train.tsv'.format(str(fold_no)), 'w') as f:
            f.writelines(['{}\n'.format(line) for line in train_lines])

        with open('folds/{}/nations_valid.tsv'.format(str(fold_no)), 'w') as f:
            f.writelines(['{}\n'.format(line) for line in valid_lines])

        with open('folds/{}/nations_test.tsv'.format(str(fold_no)), 'w') as f:
            f.writelines(['{}\n'.format(line) for line in test_lines])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
