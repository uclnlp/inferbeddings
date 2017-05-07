#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging

import sys
import os

import numpy as np

from inferbeddings.io import read_triples
from inferbeddings.knowledgebase import Fact, KnowledgeBaseParser


from inferbeddings import evaluation

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    logger.info('Command line: {}'.format(' '.join(arg for arg in argv)))

    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Baselines', formatter_class=formatter)
    argparser.add_argument('--train', '-t', required=True, action='store', type=str, default=None)
    argparser.add_argument('--valid', '-v', action='store', type=str, default=None)
    argparser.add_argument('--test', '-T', action='store', type=str, default=None)

    args = argparser.parse_args(argv)

    train_path, valid_path, test_path = args.train, args.valid, args.test

    assert train_path is not None
    pos_train_triples, _ = read_triples(train_path)

    pos_valid_triples, neg_valid_triples = read_triples(valid_path) if valid_path else (None, None)
    pos_test_triples, neg_test_triples = read_triples(test_path) if test_path else (None, None)

    def fact(s, p, o):
        return Fact(predicate_name=p, argument_names=[s, o])

    train_facts = [fact(s, p, o) for s, p, o in pos_train_triples]

    valid_facts = [fact(s, p, o) for s, p, o in pos_valid_triples] if pos_valid_triples is not None else []
    valid_facts_neg = [fact(s, p, o) for s, p, o in neg_valid_triples] if neg_valid_triples is not None else []

    test_facts = [fact(s, p, o) for s, p, o in pos_test_triples] if pos_test_triples is not None else []
    test_facts_neg = [fact(s, p, o) for s, p, o in neg_test_triples] if neg_test_triples is not None else []

    logger.info('#Training: {}, #Validation: {}, #Test: {}'
                .format(len(train_facts), len(valid_facts), len(test_facts)))

    parser = KnowledgeBaseParser(train_facts + valid_facts + test_facts)

    nb_entities = len(parser.entity_vocabulary)
    nb_predicates = len(parser.predicate_vocabulary)

    train_sequences = parser.facts_to_sequences(train_facts)

    valid_sequences = parser.facts_to_sequences(valid_facts)
    valid_sequences_neg = parser.facts_to_sequences(valid_facts_neg)

    test_sequences = parser.facts_to_sequences(test_facts)
    test_sequences_neg = parser.facts_to_sequences(test_facts_neg)

    train_triples = [(s, p, o) for (p, [s, o]) in train_sequences]

    valid_triples = [(s, p, o) for (p, [s, o]) in valid_sequences]
    valid_triples_neg = [(s, p, o) for (p, [s, o]) in valid_sequences_neg]

    test_triples = [(s, p, o) for (p, [s, o]) in test_sequences]
    test_triples_neg = [(s, p, o) for (p, [s, o]) in test_sequences_neg]

    valid_auc_roc_lst, valid_auc_pr_lst = [], []
    test_auc_roc_lst, test_auc_pr_lst = [], []

    for seed in range(10):
        random_state = np.random.RandomState(seed)

        def scoring_function(args):
            walk_inputs, entity_inputs = args[0], args[1]
            return random_state.rand(len(walk_inputs))

        valid_auc_roc, valid_auc_pr = evaluation.evaluate_auc(scoring_function,
                                                              valid_triples, valid_triples_neg,
                                                              nb_entities, nb_predicates, tag='valid')
        valid_auc_roc_lst += [valid_auc_roc]
        valid_auc_pr_lst += [valid_auc_pr]

        test_auc_roc, test_auc_pr = evaluation.evaluate_auc(scoring_function,
                                                            test_triples, test_triples_neg,
                                                            nb_entities, nb_predicates, tag='test')
        test_auc_roc_lst += [test_auc_roc]
        test_auc_pr_lst += [test_auc_pr]

    def stats(values):
        return '{0:.4f} ± {1:.4f}'.format(round(np.mean(values), 4), round(np.std(values), 4))

    logger.info('VALID AUC-ROC: {}'.format(stats(valid_auc_roc_lst)))
    logger.info('VALID AUC-PR: {}'.format(stats(valid_auc_pr_lst)))
    logger.info('TEST AUC-ROC: {}'.format(stats(test_auc_roc_lst)))
    logger.info('TEST AUC-PR: {}'.format(stats(test_auc_pr_lst)))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
