#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import os.path

import sys
import argparse
import logging


def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs])


def to_cmd(c, _path=None):
    if _path is None:
        _path = '/home/pminervi/workspace/inferbeddings/'
    loss_str = ''
    if c['loss'] == 'hinge':
        loss_str = '--loss hinge'
    elif c['loss'] == 'pairwise_hinge':
        loss_str = '--pairwise-loss hinge'
    command = 'python3 {}/bin/adv-cli.py {}' \
              ' --train {}/data/yago3_mte10_5k/yago3_mte10-train.tsv.gz' \
              ' --valid {}/data/yago3_mte10_5k/yago3_mte10-valid.tsv.gz' \
              ' --test {}/data/yago3_mte10_5k/yago3_mte10-test.tsv.gz' \
              ' --nb-epochs {}' \
              ' --lr 0.1' \
              ' --nb-batches 10' \
              ' --model {}' \
              ' --similarity {}' \
              ' --margin {}' \
              ' --embedding-size {}' \
              ' {}' \
              ' {}' \
 \
              ''.format(_path, _path, _path, _path,
                        c['epochs'],
                        c['model'], c['similarity'],
                        c['margin'], c['embedding_size'], loss_str)
    return command


def to_logfile(c, path):
    outfile = "%s/ucl_yago3_adv_v1.%s.log" % (path, summary(c))
    return outfile


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Generating experiments for the UCL cluster', formatter_class=formatter)
    argparser.add_argument('--debug', '-D', action='store_true', help='Debug flag')
    argparser.add_argument('--path', '-p', action='store', type=str, default=None, help='Path')

    args = argparser.parse_args(argv)

    hyperparameters_space_1 = dict(
        epochs=[100, 500, 1000],
        model=['DistMult', 'ComplEx'],
        similarity=['dot'],
        margin=[1, 2, 5, 10],
        embedding_size=[10, 20, 50, 100, 150, 200],
        loss=['hinge', 'pairwise_hine']
    )

    configurations = list(cartesian_product(hyperparameters_space_1))

    path = '/home/pminervi/workspace/inferbeddings/logs/schematic-memory/baselines/ucl_yago3_adv_v1/'

    # Check that we are on the UCLCS cluster first
    if os.path.exists('/home/pminervi/'):
        # If the folder that will contain logs does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)

    configurations = list(configurations)

    command_lines = set()
    for cfg in configurations:
        logfile = to_logfile(cfg, path)

        completed = False
        if os.path.isfile(logfile):
            with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = '[valid]' in content and '[test]' in content and 'AUC-PR' in content

        if not completed:
            command_line = '{} > {} 2>&1'.format(to_cmd(cfg, _path=args.path), logfile)
            command_lines |= {command_line}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines)
    nb_jobs = len(sorted_command_lines)

    header = """#!/bin/bash

#$ -cwd
#$ -S /bin/bash
#$ -o /dev/null
#$ -e /dev/null
#$ -t 1-{}
#$ -l h_vmem=4G,tmem=4G
#$ -l h_rt=1:00:00

""".format(nb_jobs)

    print(header)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        print('test $SGE_TASK_ID -eq {} && {}'.format(job_id, command_line))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
