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

RULES_KRB_PATH = 'rules.krb'


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Populate a Knowledge Base', formatter_class=formatter)

    argparser.add_argument('triples', action='store', type=str, default=None)
    argparser.add_argument('clauses', action='store', type=str, default=None)

    args = argparser.parse_args(argv)

    triples_path = args.triples
    clauses_path = args.clauses

    triples, _ = read_triples(triples_path)
    predicate_names = {p for (_, p, _) in triples}

    with open(clauses_path, 'r') as f:
        clauses_str = [line.strip() for line in f.readlines()]
    clauses = [parse_clause(clause_str) for clause_str in clauses_str]

    rule_str_lst = []
    for idx, clause in enumerate(clauses):
        head, body = clause.head, clause.body
        head_str = '\t\tfacts.{}(${}, ${})'.format(head.predicate.name, head.arguments[0].name, head.arguments[1].name)
        predicate_names |= {head.predicate.name}
        body_str = ''
        for atom in body:
            body_str += '\t\tfacts.{}(${}, ${})'.format(atom.predicate.name, atom.arguments[0].name, atom.arguments[1].name)
            predicate_names |= {atom.predicate.name}
        rule_str_lst += ['rule_{}\n\tforeach\n{}\n\tassert\n{}\n'.format(idx, body_str, head_str)]

    with open(RULES_KRB_PATH, 'w') as f:
        f.writelines('{}\n'.format(rule_str) for rule_str in rule_str_lst)

    engine = knowledge_engine.engine('.')

    for (s, p, o) in tqdm(triples):
        engine.assert_('facts', p, (s, o))

    engine.activate(os.path.splitext(os.path.basename(RULES_KRB_PATH))[0])

    materialized_triples = []
    for predicate_name in tqdm(predicate_names):
        with engine.prove_goal('facts.{}($s, $o)'.format(predicate_name)) as gen:
            for vs, plan in gen:
                materialized_triples += [(vs['s'], predicate_name, vs['o'])]

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
