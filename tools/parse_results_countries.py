#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import sys

import numpy as np
from tqdm import tqdm

import argparse
import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def stats(values):
    print(len(values))
    if len(values) != 10:
        return '0'
    return '{0:.4f} ± {1:.4f}'.format(round(np.mean(values), 4), round(np.std(values), 4))


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Parse Countries logs', formatter_class=formatter)
    argparser.add_argument('path', action='store', type=str)
    args = argparser.parse_args(argv)

    path = args.path

    path_to_valid_aucpr, path_to_test_aucpr = {}, {}

    for file_path in tqdm(glob.glob('{}/*_model=DistMult_*_unit_cube=False*.log'.format(path))):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if '[valid]' in line and 'AUC-PR' in line:
                    path_to_valid_aucpr[file_path] = line
                if '[test]' in line and 'AUC-PR' in line:
                    path_to_test_aucpr[file_path] = line

    path_set = set(path_to_valid_aucpr.keys()) & set(path_to_test_aucpr.keys())

    # Use the following for debugging:
    # path_set = set(path_to_valid_aucpr.keys()) | set(path_to_test_aucpr.keys())

    new_path_to_valid_aucprs, new_path_to_test_aucprs = {}, {}

    for path in tqdm(path_set):
        _new_path = path
        for i in range(10):
            _new_path = _new_path.replace('_seed={}'.format(i), '_seed=X')

        if _new_path not in new_path_to_valid_aucprs:
            new_path_to_valid_aucprs[_new_path] = []
        if _new_path not in new_path_to_test_aucprs:
            new_path_to_test_aucprs[_new_path] = []

        new_path_to_valid_aucprs[_new_path] += [path_to_valid_aucpr[path]]
        new_path_to_test_aucprs[_new_path] += [path_to_test_aucpr[path]]

    new_paths = set(new_path_to_valid_aucprs.keys()) & set(new_path_to_test_aucprs.keys())
    new_path_to_valid_aucpr_stats, new_path_to_test_aucpr_stats = {}, {}

    for _new_path in new_paths:
        def tfl(line):
            x, y, z = line.split()
            return float(z)

        new_path_to_valid_aucpr_stats[_new_path] = stats([tfl(l) for l in new_path_to_valid_aucprs[_new_path]])
        new_path_to_test_aucpr_stats[_new_path] = stats([tfl(l) for l in new_path_to_test_aucprs[_new_path]])

    #for p, s in new_path_to_valid_aucpr_stats.items():
    #    print('{}\t{}\t[VALID]'.format(s, p))

    #for p, s in new_path_to_test_aucpr_stats.items():
    #    print('{}\t{}\t[TEST]'.format(s, p))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
