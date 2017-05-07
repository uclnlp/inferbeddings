#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import sys

import numpy as np
#from tqdm import tqdm
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

    for file_path in glob.glob('{}/*_model=DistMult_*_unit_cube=False*.log'.format(path)):
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

    new_paths = set(new_path_to_valid_aucprs.keys()) | set(new_path_to_test_aucprs.keys())
    new_path_to_valid_aucpr_stats, new_path_to_test_aucpr_stats = {}, {}

    def stats(values):
        return '{0:.4f} ± {1:.4f}'.format(round(np.mean(values), 4), round(np.std(values), 4))\
            if len(values) == 10 else '0'

    for _new_path in new_paths:
        new_path_to_valid_aucpr_stats[_new_path] = stats([float(l.split()[2]) for l in new_path_to_valid_aucprs[_new_path]])
        new_path_to_test_aucpr_stats[_new_path] = stats([float(l.split()[2]) for l in new_path_to_test_aucprs[_new_path]])

    model_names = ['ERMLP', 'DistMult', 'ComplEx']

    name_to_regex = {}
    for model_name in model_names:
        for s in [1, 2, 3]:
            name_to_regex['{}-ASR-S{}'.format(model_name, s)] = '*_model={}*_s={}_*.log'.format(model_name, s)
            name_to_regex['{}-S{}'.format(model_name, s)] = '*_adv_weight=0_*_model={}*_s={}_*.log'.format(model_name, s)

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

    print(regex_to_best_valid)

    name_to_best_test = {}
    for regex, (path, valid_stats) in regex_to_best_valid.items():
        name = regex_to_name[regex]
        test_stats = new_path_to_test_aucpr_stats[path]
        name_to_best_test[name] = test_stats

    sorted_names = sorted(name_to_regex.keys())
    for name in sorted_names:
        best_test = name_to_best_test[name]
        print(name, best_test)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
