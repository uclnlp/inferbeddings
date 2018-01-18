#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse

import gzip
import json

import numpy as np
from tqdm import tqdm

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    def fmt(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('SNLI Subsampler', formatter_class=fmt)

    argparser.add_argument('--path', '-p', action='store', type=str, default='snli_1.0_train.jsonl.gz')
    argparser.add_argument('--seed', '-s', action='store', type=int, default=0)
    argparser.add_argument('--fraction', '-f', action='store', type=float, default=0.2)

    args = argparser.parse_args(argv)

    path = args.path
    seed = args.seed
    fraction = args.fraction

    obj_lst = []
    with gzip.open(path, 'rb') as f:
        for line in tqdm(f):
            dl = line.decode('utf-8')
            obj = json.loads(dl)
            obj_lst += [obj]

    rs = np.random.RandomState(seed)

    nb_obj = len(obj_lst)
    # Round to the closest integer
    nb_samples = int(round(nb_obj * fraction))

    sample_idxs = rs.choice(nb_obj, nb_samples, replace=False)
    sample_obj_lst = [obj_lst[i] for i in sample_idxs]

    for obj in sample_obj_lst:
        print(obj)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
