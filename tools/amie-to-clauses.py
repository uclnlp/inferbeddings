#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

import logging
import argparse

logger = logging.getLogger(os.path.basename(sys.argv[0]))


__author__ = 'pminervini'
__copyright__ = 'UCLMR'


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('AMIE+ to Horn Clauses conversion tool', formatter_class=formatter)

    # Rules-related arguments
    argparser.add_argument('logfile', type=argparse.FileType('r'), help='AMIE+ Log')

    argparser.add_argument('--std-confidence-thr', '-C', action='store', type=float, help='Threshold on Std Confidence')
    argparser.add_argument('--body-size-thr', '-B', action='store', type=float, help='Threshold on body size')
    argparser.add_argument('--head-coverage-thr', '-H', action='store', type=float, help='Threshold on Head Coverage')
    argparser.add_argument('--show-weights', '-s', action='store_true')

    args = argparser.parse_args(argv)

    logfile = args.logfile

    hc_thr = args.head_coverage_thr
    std_c_thr = args.std_confidence_thr
    body_thr = args.body_size_thr

    show_weights = args.show_weights

    predicate_features = {}

    for line in logfile:
        components = line.strip().split('\t')
        if len(components) > 1:
            rule_str = components[0].split()
            if rule_str[0] != 'Rule':
                head_lst, body_lst = rule_str[-3:], rule_str[:-4]

                def lst_to_triples(lst):
                    return [(lst[i], lst[i + 1], lst[i + 2]) for i in range(0, len(lst), 3)]

                head, body = lst_to_triples(head_lst)[0], lst_to_triples(body_lst)
                source_var, target_predicate, target_var = head[0], head[1], head[2]

                source_var_local = source_var
                hops = []

                while len(body) > 0:
                    logging.debug(head, body)

                    hop = None
                    for i in range(len(body)):
                        if hop is None:
                            triple = body[i]

                            if triple[0] == source_var_local:
                                hop = {"predicate": triple[1], "reverse": False}
                                source_var_local = triple[2]
                                del body[i]

                            elif triple[2] == source_var_local:
                                hop = {"predicate": triple[1], "reverse": True}
                                source_var_local = triple[0]
                                del body[i]

                    hops += [hop]

                feature = {"hops": hops}
                head_coverage = float(components[1])
                std_confidence = float(components[2])
                pca_confidence = float(components[3])
                positive_examples = float(components[4])
                body_size = float(components[5])
                pca_body_size = float(components[6])

                if target_predicate not in predicate_features:
                    predicate_features[target_predicate] = []

                obj = {
                    'feature': feature,
                    'head_coverage': head_coverage,
                    'std_confidence': std_confidence,
                    'pca_confidence': pca_confidence,
                    'positive_examples': positive_examples,
                    'body_size': body_size,
                    'pca_body_size': pca_body_size
                }

                predicate_features[target_predicate] += [obj]

                logger.debug(target_predicate, [str(hop) for hop in hops], components[1])

    obj = []
    for predicate, features in predicate_features.items():
        obj += [{"predicate": predicate, "features": features}]

    for rule in obj:
        head_predicate = rule['predicate']

        features = rule['features']
        for _feature in features:
            head_coverage = _feature['head_coverage']
            std_confidence = _feature['std_confidence']
            body_size = _feature['body_size']

            feature = _feature['feature']
            hops = feature['hops']

            clause_body, last_i = '', None
            for i, hop in enumerate(hops):
                first_var = 'X{}'.format((i + 1) if hop['reverse'] else i)
                second_var = 'X{}'.format(i if hop['reverse'] else (i + 1))
                clause_body += (', ' if len(clause_body) > 0 else '') + ('{}({}, {})'.format(hop['predicate'], first_var, second_var))
                last_i = i

            clause_head = '{}({}, {})'.format(head_predicate, 'X{}'.format(0), 'X{}'.format((last_i + 1)))
            clause = '{} :- {}'.format(clause_head, clause_body)

            if hc_thr is None or head_coverage >= hc_thr:
                if std_c_thr is None or std_confidence >= std_c_thr:
                    if body_thr is None or body_size >= body_thr:
                        print(('{}\t{}\t{}'.format(clause, head_coverage, std_confidence)) if show_weights else '{}'.format(clause))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
