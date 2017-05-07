#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import sys

import numpy as np
from tqdm import tqdm
import fnmatch

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

    def stats(values):
        return '{0:.4f} ± {1:.4f}'.format(round(np.mean(values), 4), round(np.std(values), 4))\
            if len(values) == 10 else '0'

    for _new_path in new_paths:
        new_path_to_valid_aucpr_stats[_new_path] = stats([float(l.split()[2]) for l in new_path_to_valid_aucprs[_new_path]])
        new_path_to_test_aucpr_stats[_new_path] = stats([float(l.split()[2]) for l in new_path_to_test_aucprs[_new_path]])

    name_to_regex = {
        'ADistMult-S1': '*_model=DistMult*_s=1_*.log',
        'DistMult-S1': '*_adv_weight=0_*_model=DistMult*_s=1_*.log',

        'ADistMult-S2': '*_model=DistMult*_s=2_*.log',
        'DistMult-S2': '*_adv_weight=0_*_model=DistMult*_s=2_*.log',

        'ADistMult-S3': '*_model=DistMult*_s=3_*.log',
        'DistMult-S3': '*_adv_weight=0_*_model=DistMult*_s=3_*.log',

        #'AComplEx-S1': '*_model=ComplEx_*.log',
        #'ComplEx-S1': '*_adv_weight=0_*_model=ComplEx_*.log'
    }
    regex_to_name = {regex: name for name, regex in name_to_regex.items()}

    regex_to_best_valid = {regex: None for _, regex in name_to_regex.items()}

    for path, stats in new_path_to_valid_aucpr_stats.items():
        for regex, best_valid in regex_to_best_valid.items():
            if fnmatch.fnmatch(path, regex):
                if best_valid is None:
                    regex_to_best_valid[regex] = (path, stats)
                else:
                    (_, best_stats) = best_valid
                    if float(stats.split(' ')[0]) > float(best_stats.split(' ')[0]):
                        regex_to_best_valid[regex] = (path, stats)

    name_to_best_test = {}
    for regex, (path, valid_stats) in regex_to_best_valid.items():
        name = regex_to_name[regex]
        test_stats = new_path_to_test_aucpr_stats[path]
        name_to_best_test[name] = test_stats

    for name, best_test in name_to_best_test.items():
        print(name, best_test)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
