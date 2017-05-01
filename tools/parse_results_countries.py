#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import sys

from tqdm import tqdm

import argparse
import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Parse Countries logs', formatter_class=formatter)
    argparser.add_argument('path', action='store', type=str)
    args = argparser.parse_args(argv)

    path = args.path

    path_to_valid_aucpr, path_to_test_aucpr = {}, {}

    for file_path in tqdm(glob.glob('{}/*.log'.format(path))):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if '[valid]' in line and 'AUC-PR' in line:
                    path_to_valid_aucpr[file_path] = line
                if '[test]' in line and 'AUC-PR' in line:
                    path_to_test_aucpr[file_path] = line

    path_set = set(path_to_valid_aucpr.keys()) | set(path_to_test_aucpr.keys())

    new_path_to_valid_aucprs, new_path_to_test_aucprs = {}, {}

    for path in path_set:
        _new_path = path
        for i in range(10):
            _new_path = _new_path.replace('_seed={}'.format(i), '_seed=X')

        if _new_path not in new_path_to_valid_aucprs:
            new_path_to_valid_aucprs[_new_path] = []
        if _new_path not in new_path_to_test_aucprs:
            new_path_to_test_aucprs[_new_path] = []

        new_path_to_valid_aucprs[_new_path] += [path_to_valid_aucpr[path]]
        new_path_to_test_aucprs[_new_path] += [path_to_test_aucpr[path]]



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
