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


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Generate plots for X-Shot Learning', formatter_class=formatter)

    argparser.add_argument('base', action='store', type=str, default='./',
                           help='Folder where "inferbeddings" is located')
    argparser.add_argument('logs', action='store', type=str,
                           help='Folder where the logs are located (e.g. "logs/ucl_wn18_adv_xshot_v1/")')

    args = argparser.parse_args(argv)

    assert args.base is not None
    assert args.logs is not None

    if args.base is not None:
        os.chdir(args.base)

    for model_name in ['TransE', 'DistMult', 'ComplEx']:
        for sample_size in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:

            mr = []
            mrr = []

            hits_at_1 = []
            hits_at_3 = []
            hits_at_5 = []
            hits_at_10 = []

            similarity_name = 'dot'

            cmd = './tools/parse_results_filtered.sh ' \
                  '{}/*model={}*similarity={}*subsample_size={}.log'.format(args.logs, model_name, similarity_name, sample_size)

            mr += [float(exec(cmd + '| grep MR: | awk \'{ print $6 }\'').strip())]
            mrr += [float(exec(cmd + '| grep MRR: | awk \'{ print $6 }\'').strip())]

            hits_at_1 += [float(exec(cmd + '| grep Hits@1: | awk \'{ print $6 }\'').strip())]
            hits_at_3 += [float(exec(cmd + '| grep Hits@3: | awk \'{ print $6 }\'').strip())]
            hits_at_5 += [float(exec(cmd + '| grep Hits@5: | awk \'{ print $6 }\'').strip())]
            hits_at_10 += [float(exec(cmd + '| grep Hits@10: | awk \'{ print $6 }\'').strip())]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
