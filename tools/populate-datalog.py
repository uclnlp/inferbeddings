#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse

from pyDatalog import pyDatalog

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

    # Create a set containing all the entities from the triples
    entity_names = {s for (s, _, _) in triples} | {o for (_, _, o) in triples}

    # Create a set containing all predicate names from the triples and clauses
    predicate_names = {p for (_, p, _) in triples}
    for clause in clauses:
        predicate_names |= {clause.head.predicate.name}
        for atom in clause.body:
            predicate_names |= {atom.predicate.name}

    # Associate each entity and predicate to an unique index
    entity_to_idx = {entity: idx for idx, entity in enumerate(entity_names)}
    idx_to_entity = {idx: entity for entity, idx in entity_to_idx.items()}

    predicate_to_idx = {predicate: idx for idx, predicate in enumerate(predicate_names)}
    idx_to_predicate = {idx: predicate for predicate, idx in predicate_to_idx.items()}

    logger.info('Asserting facts ..')

    # Asserting the facts
    for (s, p, o) in triples:
        pyDatalog.assert_fact('p', entity_to_idx[s], predicate_to_idx[p], entity_to_idx[o])

    logger.info('Querying triples ..')

    ans = pyDatalog.ask('p(S, P, O)')
    print(len(ans.answers))

    logger.info('Loading rules ..')

    def atom_to_str(atom):
        atom_predicate_idx = predicate_to_idx[atom.predicate.name]
        atom_arg_0, atom_arg_1 = atom.arguments[0], atom.arguments[1]
        return 'p({}, {}, {})'.format(atom_arg_0, atom_predicate_idx, atom_arg_1)

    def clause_to_str(clause):
        head, body = clause.head, clause.body
        return '{} <= {}'.format(atom_to_str(head), ' & '.join([atom_to_str(a) for a in body]))

    rules_str = '\n'.join([clause_to_str(clause) for clause in clauses])

    pyDatalog.load(rules_str)

    logger.info('Querying triples ..')

    ans = pyDatalog.ask('p(S, P, O)')
    answers = sorted(ans.answers)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
