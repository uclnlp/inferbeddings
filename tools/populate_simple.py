#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse

from pyke import knowledge_engine
from tqdm import tqdm

from inferbeddings.io import read_triples
from inferbeddings.parse import parse_clause

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Populate a Knowledge Base', formatter_class=formatter)

    argparser.add_argument('triples', action='store', type=str, default=None)
    argparser.add_argument('clauses', action='store', type=str, default=None)
    argparser.add_argument('--output', '-o', action='store', type=str, default=None)

    args = argparser.parse_args(argv)

    triples_path = args.triples
    clauses_path = args.clauses
    output_path = args.output

    triples, _ = read_triples(triples_path)

    # Parse the clauses using Sebastian's parser
    with open(clauses_path, 'r') as f:
        clauses_str = [line.strip() for line in f.readlines()]
    clauses = [parse_clause(clause_str) for clause_str in clauses_str]

    # Create a set containing all predicate names
    predicate_names = {p for (_, p, _) in triples}
    for clause in clauses:
        predicate_names |= {clause.head.predicate.name}
        for atom in clause.body:
            predicate_names |= {atom.predicate.name}

    # The original predicate names might not be handled well by Pyke (it's the case of e.g. Freebase)
    # Replace them with p1, p2, p3 etc.
    predicate_to_idx = {predicate: 'p{}'.format(idx) for idx, predicate in enumerate(predicate_names)}
    idx_to_predicate = {idx: predicate for predicate, idx in predicate_to_idx.items()}


