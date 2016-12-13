#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import logging
import argparse

import json


__author__ = 'pminervini'
__copyright__ = 'INSIGHT Centre for Data Analytics 2016'


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('AMIE+ to JSON conversion tool', formatter_class=formatter)

    # Rules-related arguments
    argparser.add_argument('logfile', type=argparse.FileType('r'), help='AMIE+ Log')

    args = argparser.parse_args(argv)
    logfile = args.logfile

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
                                #hop = domain.Hop(triple[1], is_inverse=False)
                                hop = {"predicate": triple[1], "reverse": False}
                                source_var_local = triple[2]
                                del body[i]

                            elif triple[2] == source_var_local:
                                #hop = #domain.Hop(triple[1], is_inverse=True)
                                hop = {"predicate": triple[1], "reverse": True}
                                source_var_local = triple[0]
                                del body[i]

                    hops += [hop]

                #feature = domain.Feature(hops)
                feature = {"hops": hops}
                weight = float(components[1])

                if target_predicate not in predicate_features:
                    predicate_features[target_predicate] = []

                predicate_features[target_predicate] += [{"feature": feature, "weight": weight}]

                logging.debug(target_predicate, [str(hop) for hop in hops], components[1])

    obj = []
    for predicate, features in predicate_features.items():
        obj += [{"predicate": predicate, "features": features}]

    json_str = json.dumps(obj, sort_keys=True, indent=2)
    print(json_str)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
