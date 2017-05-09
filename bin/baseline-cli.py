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

    argparser.add_argument('--model', '-m', action='store', type=str, choices=['random', 'frequency'], default=None)
    argparser.add_argument('--nb-runs', '-r', action='store', type=int, default=10)

    args = argparser.parse_args(argv)

    train_path, valid_path, test_path = args.train, args.valid, args.test

    model_name = args.model
    nb_runs = args.nb_runs

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

    region_lst = 'africa americas asia europe oceania'.split(' ')
    subregion_lst = 'australia_and_new_zealand caribbean central_america central_asia central_europe eastern_africa eastern_asia eastern_europe melanesia micronesia middle_africa northern_africa northern_america northern_europe polynesia south-eastern_asia south_america southern_africa southern_asia southern_europe western_africa western_asia western_europe'.split(' ')

    locatedIn_idx = None
    if 'locatedin' in parser.predicate_to_index:
        locatedIn_idx = parser.predicate_to_index['locatedin']
    elif 'locatedIn' in parser.predicate_to_index:
        locatedIn_idx = parser.predicate_to_index['locatedIn']
    assert locatedIn_idx is not None

    region_idx_lst = [parser.entity_to_index[region] for region in region_lst]
    subregion_idx_lst = [parser.entity_to_index[subregion] for subregion in subregion_lst]

    region_idx_to_frequency = {region_idx: 0 for region_idx in region_idx_lst}
    for s, p, o in train_triples:
        if s not in subregion_idx_lst and p == locatedIn_idx and o in region_idx_lst:
            region_idx_to_frequency[o] += 1

    most_frequent_region_idx = sorted(region_idx_to_frequency.items(), key=lambda x: x[1])[0]

    for seed in range(nb_runs):
        random_state = np.random.RandomState(seed)

        def random_scoring_function(args):
            walk_inputs, entity_inputs = args[0], args[1]
            return random_state.rand(len(walk_inputs))

        def frequency_scoring_function(args):
            walk_inputs, entity_inputs = args[0], args[1]
            nb_instances = len(walk_inputs)

            res = np.zeros(nb_instances)
            for idx in range(nb_instances):
                assert walk_inputs[idx][0] == locatedIn_idx
                region_idx = entity_inputs[idx][1]
                res[idx] = region_idx_to_frequency[region_idx]
            print(res)
            return res

        scoring_function = None
        if model_name == 'random':
            scoring_function = random_scoring_function
        elif model_name == 'frequency':
            scoring_function = frequency_scoring_function

        assert scoring_function is not None

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

    logger.info('[VALID]\tAUC-ROC: {}\tAUC-PR: {}'.format(stats(valid_auc_roc_lst), stats(valid_auc_pr_lst)))
    logger.info('[TEST]\tAUC-ROC: {}\tAUC-PR: {}'.format(stats(test_auc_roc_lst), stats(test_auc_pr_lst)))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
