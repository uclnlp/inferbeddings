#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sample usage: $ ./bin/loss-cli.py -m models/wn18/wn18_v1.pkl -c data/wn18/clauses/clauses_0.9.pl
"""

import argparse
import pickle

from inferbeddings.parse import parse_clause

import os
import sys
import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))




def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Rule-based Ground Loss', formatter_class=formatter)

    argparser.add_argument('--model', '-m', action='store', type=str, required=True, help='Trained model')
    argparser.add_argument('--clauses', '-c', action='store', type=str, required=True, help='Horn clauses')

    args = argparser.parse_args(argv)

    model_path = args.model
    assert model_path is not None

    clauses_path = args.clauses
    assert clauses_path is not None

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(clauses_path, 'r') as f:
        clauses = [parse_clause(line.strip()) for line in f.readlines()]

    entities = model['entity_to_index'].keys()
    predicates = model['predicate_to_index'].keys()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
