#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sample usage:
$ mkdir tmp ; cd tmp
$ PYTHONPATH=. ../tools/populate.py ../data/wn18/wordnet-mlj12-valid.txt ../data/wn18/clauses/clauses_0.9.pl -o /dev/stdout
"""

import os
import sys

import argparse

from pyke import knowledge_engine
from tqdm import tqdm

from inferbeddings.io import read_triples
from inferbeddings.parse import parse_clause

import logging

RULES_KRB_PATH = 'rules.krb'


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

    # Generate a Pyke rule base for reasoning via forward chaining
    rule_str_lst = []
    for idx, clause in enumerate(clauses):
        head, body = clause.head, clause.body
        head_str = '\t\tfacts.{}(${}, ${})'.format(predicate_to_idx[head.predicate.name], head.arguments[0].name, head.arguments[1].name)
        body_str = ''
        for atom in body:
            body_str += '\t\tfacts.{}(${}, ${})\n'.format(predicate_to_idx[atom.predicate.name], atom.arguments[0].name, atom.arguments[1].name)
        rule_str_lst += ['rule_{}\n\tforeach\n{}\n\tassert\n{}\n'.format(idx, body_str, head_str)]

    # Write the Pyke rule base to file
    with open(RULES_KRB_PATH, 'w') as f:
        f.writelines('{}\n'.format(rule_str) for rule_str in rule_str_lst)

    engine = knowledge_engine.engine('.')

    # Assert starting facts, corresponding to the triples already in the Knowledge Graph
    for (s, p, o) in tqdm(triples):
        engine.assert_('facts', predicate_to_idx[p], (s, o))

    engine.activate(os.path.splitext(os.path.basename(RULES_KRB_PATH))[0])

    # For each predicate p, query the reasoning engine ..
    materialized_triples = []
    for predicate_name in tqdm(predicate_names):
        # .. asking for all subject s and object o pairs such that (s, p, o) is entailed by the knowledge base
        with engine.prove_goal('facts.{}($s, $o)'.format(predicate_to_idx[predicate_name])) as gen:
            for vs, plan in gen:
                materialized_triples += [(vs['s'], predicate_name, vs['o'])]

    if output_path is not None:
        # Write the materialized triples to file
        with open(output_path, 'w') as f:
            f.writelines('{}\t{}\t{}\n'.format(s, p, o) for s, p, o in materialized_triples)


if __name__ == '__main__':
    sys.setrecursionlimit(65536)
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
