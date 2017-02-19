#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse

import subprocess

import logging


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Generate plots for X-Shot Learning', formatter_class=formatter)

    argparser.add_argument('--base', action='store', type=str,
                           default='/home/pasquale/ucl/workspace/inferbeddings',
                           help='Folder where "inferbeddings" is located')

    args = argparser.parse_args(argv)

    if args.base is not None:
        os.chdir(args.base)

    for sample_size in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        cmd = './tools/parse_results_filtered.sh ' \
              'logs/ucl_fb15k_adv_v1/*_subsample_size={}.log'.format(sample_size)
        p = subprocess.Popen(['sh', '-c', './tools/parse_results_filtered.sh logs/ucl_fb15k_adv_v1/*.log'],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
