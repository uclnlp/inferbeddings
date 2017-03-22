#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import argparse

from inferbeddings.io import read_triples
from inferbeddings.knowledgebase import Fact, KnowledgeBaseParser
from inferbeddings.parse import parse_clause
from inferbeddings.logic import materialize

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)
    argparser = argparse.ArgumentParser('Generates a Test-I/Test-II/Test-ALL test sets', formatter_class=formatter)

    argparser.add_argument('train', action='store', type=str, default=None)
    argparser.add_argument('valid', action='store', type=str, default=None)
    argparser.add_argument('test', action='store', type=str, default=None)
    argparser.add_argument('clauses', action='store', type=str, default=None)
    argparser.add_argument('--output-type', '-t', type=int, default=1, choices=[1, 2])

    args = argparser.parse_args(argv)

    train_path, valid_path, test_path = args.train, args.valid, args.test
    output_type = args.output_type

    train_triples, _ = read_triples(train_path)
    valid_triples, _ = read_triples(valid_path)
    test_triples, _ = read_triples(test_path)

    def fact(s, p, o):
        return Fact(predicate_name=p, argument_names=[s, o])

    train_facts = [fact(s, p, o) for s, p, o in train_triples]
    valid_facts = [fact(s, p, o) for s, p, o in valid_triples]
    test_facts = [fact(s, p, o) for s, p, o in test_triples]

    parser = KnowledgeBaseParser(train_facts + valid_facts + test_facts)

    clauses_path = args.clauses
    with open(clauses_path, 'r') as f:
        clauses = [parse_clause(line.strip()) for line in f.readlines()]

    for clause in clauses:
        logging.info('Clause: {}'.format(clause))

    # Put all triples in the form of sets of tuples
    train_triples = {(fact.argument_names[0], fact.predicate_name, fact.argument_names[1]) for fact in train_facts}
    valid_triples = {(fact.argument_names[0], fact.predicate_name, fact.argument_names[1]) for fact in valid_facts}
    test_triples = {(fact.argument_names[0], fact.predicate_name, fact.argument_names[1]) for fact in test_facts}

    m_train_facts = materialize(train_facts, clauses, parser)
    m_train_triples = {(fact.argument_names[0], fact.predicate_name, fact.argument_names[1]) for fact in m_train_facts}

    # Check if the sets of triples are non-empty
    assert len(train_triples) > 0
    assert len(valid_triples) > 0
    assert len(test_triples) > 0
    assert len(m_train_triples) > len(train_triples)

    # Check that their intersections are empty (e.g. no test triple appear in the training set etc.)
    assert len(train_triples & valid_triples) == 0
    assert len(train_triples & test_triples) == 0
    assert len(valid_triples & test_triples) == 0

    # Note that some of the test triples can be inferred by directly applying these rules on the training set
    # (pure logical inference). On each dataset, we further split the test set into two parts, test-I and test-II.
    # The former contains triples that cannot be directly inferred by pure logical inference, and the latter the
    # remaining test triples. Table 3 gives some statistics of the datasets, including the number of entities,
    # relations, triples in training/validation/test-I/test-II set, and ground rules.
    assert output_type in {1, 2}

    # Triples that cannot be directly inferred by pure logical inference
    test_1_triples = test_triples - m_train_triples

    # Triples that can be directly inferred by pure logical inference
    test_2_triples = test_triples & m_train_triples

    nb_1_triples, nb_2_triples, nb_all_triples = len(test_1_triples), len(test_2_triples), len(test_triples)
    assert nb_1_triples + nb_2_triples == nb_all_triples
    assert len(test_1_triples | test_2_triples) == nb_all_triples

    logger.info('#Test-I: {}, #Test-II: {}, #Test-ALL: {}'.format(nb_1_triples, nb_2_triples, nb_all_triples))

    print(test_triples & m_train_triples)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
