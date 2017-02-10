#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

import logging


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Plot Embeddings', formatter_class=formatter)
    argparser.add_argument('data-path', action='store', type=str, required=True)

    args = argparser.parse_args(argv)

    data_path = args.data_path

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    print(data.keys())

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])