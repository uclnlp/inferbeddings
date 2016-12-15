#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse

from pyke import knowledge_engine
from pyke import krb_traceback
from pyke import goal

from inferbeddings.io import read_triples
from inferbeddings.parse import parse_clause

import logging

RULES_KRB_PATH = './rules.krb'


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

    with open(clauses_path, 'r') as f:
        clauses_str = [line.strip() for line in f.readlines()]
    clauses = [parse_clause(clause_str) for clause_str in clauses_str]

    def to_rule_str(idx, clause):
        head, body = clause.head, clause.body
        head_str = '\t\tfacts.{}(${}, ${})'.format(head.predicate.name, head.arguments[0].name, head.arguments[1].name)
        body_str = ''
        for atom in body:
            body_str += '\t\tfacts.{}(${}, ${})'.format(atom.predicate.name, head.arguments[0].name, atom.arguments[1].name)
        return 'rule_{}\n\tforeach\n{}\n\tassert\n{}\n'.format(idx, body_str, head_str)

    rule_str_lst = [to_rule_str(idx, clause) for idx, clause in enumerate(clauses)]

    with open(RULES_KRB_PATH, 'w') as f:
        f.writelines('%s\n'.format(rule_str) for rule_str in rule_str_lst)

    os.remove(RULES_KRB_PATH)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
