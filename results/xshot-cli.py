#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse

import subprocess

import logging


def exec(cmd):
    p = subprocess.Popen(['sh', '-c', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return out


def get_results(cmd):
    mr = float(exec(cmd + '| grep MR: | awk \'{ print $6 }\'').strip())
    mrr = float(exec(cmd + '| grep MRR: | awk \'{ print $6 }\'').strip())
    hits_at_1 = float(exec(cmd + '| grep Hits@1: | awk \'{ print $6 }\'').strip())
    hits_at_3 = float(exec(cmd + '| grep Hits@3: | awk \'{ print $6 }\'').strip())
    hits_at_5 = float(exec(cmd + '| grep Hits@5: | awk \'{ print $6 }\'').strip())
    hits_at_10 = float(exec(cmd + '| grep Hits@10: | awk \'{ print $6 }\'').strip())
    return mr, mrr, hits_at_1, hits_at_3, hits_at_5, hits_at_10


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Generate plots for X-Shot Learning', formatter_class=formatter)

    argparser.add_argument('base', action='store', type=str, default='./',
                           help='Folder where "inferbeddings" is located')
    argparser.add_argument('logs', action='store', type=str,
                           help='Folder where the logs are located (e.g. "logs/ucl_wn18_adv_xshot_v1/")')
    argparser.add_argument('logic', action='store', type=str,
                           help='Folder where the logic logs are located (e.g. "logs/ucl_wn18_adv_xshot_v1/")')

    args = argparser.parse_args(argv)

    assert args.base is not None
    assert args.logs is not None
    assert args.logic is not None

    if args.base is not None:
        os.chdir(args.base)

    model_to_adversarial_results = {}
    model_to_standard_results = {}
    model_to_naacl_results = {}
    model_to_logic_results = {}

    for model_name in ['TransE', 'DistMult', 'ComplEx']:

        # Adversarial results

        mr, mrr = [], []
        hits_at_1, hits_at_3, hits_at_5, hits_at_10 = [], [], [], []

        for sample_size in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            similarity_name = 'dot'
            cmd = './tools/parse_results_filtered.sh ' \
                  '{}/*model={}*similarity={}*subsample_size={}.log'.format(args.logs, model_name, similarity_name, sample_size)

            _mr, _mrr, _hits_at_1, _hits_at_3, _hits_at_5, _hits_at_10 = get_results(cmd)

            mr += [_mr]
            mrr += [_mrr]

            hits_at_1 += [_hits_at_1]
            hits_at_3 += [_hits_at_3]
            hits_at_5 += [_hits_at_5]
            hits_at_10 += [_hits_at_10]

        results = {
            'mr': mr, 'mrr': mrr,
            'hits_at_1': hits_at_1, 'hits_at_3': hits_at_3,
            'hits_at_5': hits_at_5, 'hits_at_10': hits_at_10
        }

        model_to_adversarial_results[model_name] = results

        # Standard results (no Adversarial)

        mr, mrr = [], []
        hits_at_1, hits_at_3, hits_at_5, hits_at_10 = [], [], [], []

        for sample_size in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            similarity_name = 'dot'
            cmd = './tools/parse_results_filtered.sh ' \
                  '{}/*adv_weight=0_*model={}*similarity={}*subsample_size={}.log'.format(args.logs, model_name, similarity_name, sample_size)

            _mr, _mrr, _hits_at_1, _hits_at_3, _hits_at_5, _hits_at_10 = get_results(cmd)

            mr += [_mr]
            mrr += [_mrr]

            hits_at_1 += [_hits_at_1]
            hits_at_3 += [_hits_at_3]
            hits_at_5 += [_hits_at_5]
            hits_at_10 += [_hits_at_10]

        results = {
            'mr': mr, 'mrr': mrr,
            'hits_at_1': hits_at_1, 'hits_at_3': hits_at_3,
            'hits_at_5': hits_at_5, 'hits_at_10': hits_at_10
        }

        model_to_standard_results[model_name] = results

        # NAACL results

        mr, mrr = [], []
        hits_at_1, hits_at_3, hits_at_5, hits_at_10 = [], [], [], []

        for sample_size in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            similarity_name = 'dot'
            cmd = './tools/parse_results_filtered.sh ' \
                  '{}/*adv_epochs=0_*model={}*similarity={}*subsample_size={}.log'.format(args.logs, model_name, similarity_name, sample_size)

            _mr, _mrr, _hits_at_1, _hits_at_3, _hits_at_5, _hits_at_10 = get_results(cmd)

            mr += [_mr]
            mrr += [_mrr]

            hits_at_1 += [_hits_at_1]
            hits_at_3 += [_hits_at_3]
            hits_at_5 += [_hits_at_5]
            hits_at_10 += [_hits_at_10]

        results = {
            'mr': mr, 'mrr': mrr,
            'hits_at_1': hits_at_1, 'hits_at_3': hits_at_3,
            'hits_at_5': hits_at_5, 'hits_at_10': hits_at_10
        }

        model_to_naacl_results[model_name] = results

        # Logic results

        mr, mrr = [], []
        hits_at_1, hits_at_3, hits_at_5, hits_at_10 = [], [], [], []

        for sample_size in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            similarity_name = 'dot'
            cmd = './tools/parse_results_filtered.sh ' \
                  '{}/*model={}*similarity={}*subsample_size={}.log'.format(args.logic, model_name, similarity_name, sample_size)

            _mr, _mrr, _hits_at_1, _hits_at_3, _hits_at_5, _hits_at_10 = get_results(cmd)

            mr += [_mr]
            mrr += [_mrr]

            hits_at_1 += [_hits_at_1]
            hits_at_3 += [_hits_at_3]
            hits_at_5 += [_hits_at_5]
            hits_at_10 += [_hits_at_10]

        results = {
            'mr': mr, 'mrr': mrr,
            'hits_at_1': hits_at_1, 'hits_at_3': hits_at_3,
            'hits_at_5': hits_at_5, 'hits_at_10': hits_at_10
        }

        model_to_logic_results[model_name] = results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
