#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse

import subprocess

import logging

import json

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def exec(cmd):
    p = subprocess.Popen(['sh', '-c', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return out


def _get_results(cmd):
    exec(cmd + ' > /tmp/make_america_great_again.log 2>&1')
    mr = float(exec('cat /tmp/make_america_great_again.log | grep MR: | awk \'{ print $6 }\'').strip())
    mrr = float(exec('cat /tmp/make_america_great_again.log | grep MRR: | awk \'{ print $6 }\'').strip())
    hits_at_1 = float(exec('cat /tmp/make_america_great_again.log | grep Hits@1: | tr -d "%" | awk \'{ print $6 }\'').strip())
    hits_at_3 = float(exec('cat /tmp/make_america_great_again.log | grep Hits@3: | tr -d "%" | awk \'{ print $6 }\'').strip())
    hits_at_5 = float(exec('cat /tmp/make_america_great_again.log | grep Hits@5: | tr -d "%" | awk \'{ print $6 }\'').strip())
    hits_at_10 = float(exec('cat /tmp/make_america_great_again.log | grep Hits@10: | tr -d "%" | awk \'{ print $6 }\'').strip())
    return mr, mrr, hits_at_1, hits_at_3, hits_at_5, hits_at_10


def get_results(logs, model_name, prefix):
    mr, mrr = [], []
    hits_at_1, hits_at_3, hits_at_5, hits_at_10 = [], [], [], []

    for sample_size in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        print(sample_size)
        similarity_name = 'dot'
        cmd = './tools/parse_results_filtered.sh ' \
              '{}/{}*model={}*similarity={}*subsample_size={}.log'.format(logs, prefix, model_name,
                                                                          similarity_name, sample_size)
        _mr, _mrr, _hits_at_1, _hits_at_3, _hits_at_5, _hits_at_10 = _get_results(cmd)

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

    return results


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
        print(model_name)

        # Adversarial training
        model_to_adversarial_results[model_name] = get_results(logs=args.logs, model_name=model_name, prefix='*')
        logger.info('{}: {}'.format(model_name, str(model_to_adversarial_results[model_name])))

        # Standard results (no Adversarial)
        model_to_standard_results[model_name] = get_results(logs=args.logs, model_name=model_name, prefix='*adv_weight=0_*')
        logger.info('{}: {}'.format(model_name, str(model_to_standard_results[model_name])))

        # NAACL results
        model_to_naacl_results[model_name] = get_results(logs=args.logs, model_name=model_name, prefix='*adv_epochs=0_*')
        logger.info('{}: {}'.format(model_name, str(model_to_naacl_results[model_name])))

        # Logic results
        model_to_logic_results[model_name] = get_results(logs=args.logic, model_name=model_name, prefix='*')
        logger.info('{}: {}'.format(model_name, str(model_to_logic_results[model_name])))

    with open("./results/model_to_adversarial_results.json", "w") as f:
        json.dump(model_to_adversarial_results, f, indent=2)
        f.close()
    with open("./results/model_to_standard_results.json", "w") as f:
        json.dump(model_to_standard_results, f, indent=2)
        f.close()
    with open("./results/model_to_naacl_results.json", "w") as f:
        json.dump(model_to_naacl_results, f, indent=2)
        f.close()
    with open("./results/model_to_logic_results.json", "w") as f:
        json.dump(model_to_logic_results, f, indent=2)
        f.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
