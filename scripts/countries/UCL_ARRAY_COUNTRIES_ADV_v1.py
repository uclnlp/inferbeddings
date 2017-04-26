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
    unit_cube_str = '--unit-cube' if c['unit_cube'] else ''
    loss_str = ''
    if c['loss'] == 'hinge':
        loss_str = '--loss hinge'
    elif c['loss'] == 'pairwise_hinge':
        loss_str = '--pairwise-loss hinge'
    hidden_str = '--hidden-size {}'.format(c['hidden_size']) if 'hidden_size' in c else ''
    command = 'python3 {}/bin/adv-cli.py --auc --nb-batches 1' \
              ' --train {}/data/countries/s{}/s{}_train.tsv' \
              ' --valid {}/data/countries/s{}/s{}_valid.tsv' \
              ' --test {}/data/countries/s{}/s{}_test.tsv' \
              ' --clauses {}/data/countries/clauses/s{}.pl' \
              ' --nb-epochs {}' \
              ' --lr 0.1' \
              ' --nb-batches 10' \
              ' --model {}' \
              ' --similarity {}' \
              ' --margin {}' \
              ' --embedding-size {}' \
              ' {}' \
              ' {}' \
              ' {}' \
              ' --adv-lr {} --adv-init-ground --adversary-epochs {}' \
              ' --discriminator-epochs {} --adv-weight {} --adv-batch-size {} --adv-pooling {}' \
              ''.format(_path,
                        _path, c['s'], c['s'],
                        _path, c['s'], c['s'],
                        _path, c['s'], c['s'],
                        _path, c['s'],
                        c['epochs'],
                        c['model'], c['similarity'],
                        c['margin'], c['embedding_size'], hidden_str,
                        loss_str,
                        unit_cube_str, c['adv_lr'], c['adv_epochs'],
                        c['disc_epochs'], c['adv_weight'], c['adv_batch_size'], c['adv_pooling'])
    return command


def to_logfile(c, path):
    outfile = "%s/ucl_countries_adv_v1.%s.log" % (path, summary(c))
    return outfile


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Generating experiments for the UCL cluster', formatter_class=formatter)
    argparser.add_argument('--debug', '-D', action='store_true', help='Debug flag')
    argparser.add_argument('--path', '-p', action='store', type=str, default=None, help='Path')

    args = argparser.parse_args(argv)

    hyperparameters_space_1 = dict(
        epochs=[100],
        model=['ERMLP'],
        similarity=['dot'],
        margin=[1, 2, 5, 10],
        embedding_size=[10, 20, 50, 100],
        loss=['hinge', 'pairwise_hinge'],
        adv_lr=[.1],
        adv_epochs=[0, 10],
        disc_epochs=[10],
        adv_weight=[0, 1, 100, 10000, 1000000],
        adv_batch_size=[1, 10, 100],
        adv_pooling=['max'],
        hidden_size=[1, 10, 20, 50, 100],
        unit_cube=[True, False],
        s=[1, 2, 3]
    )

    hyperparameters_space_2 = dict(
        epochs=[100],
        model=['DistMult', 'ComplEx'],
        similarity=['dot'],
        margin=[1, 2, 5, 10],
        embedding_size=[10, 20, 50, 100],
        loss=['hinge', 'pairwise_hinge'],
        adv_lr=[.1],
        adv_epochs=[0, 10],
        disc_epochs=[10],
        adv_weight=[0, 1, 100, 10000, 1000000],
        adv_batch_size=[1, 10, 100],
        adv_pooling=['max'],
        unit_cube=[True, False],
        s=[1, 2, 3]
    )

    hyperparameters_space_3 = dict(
        epochs=[100],
        model=['TransE'],
        similarity=['l1', 'l2'],
        margin=[1, 2, 5, 10],
        embedding_size=[10, 20, 50, 100],
        loss=['pairwise_hinge'],
        adv_lr=[.1],
        adv_epochs=[0, 10],
        disc_epochs=[10],
        adv_weight=[0, 1, 100, 10000, 1000000],
        adv_batch_size=[1, 10, 100],
        adv_pooling=['max'],
        unit_cube=[True, False],
        s=[1, 2, 3]
    )

    configurations = list(cartesian_product(hyperparameters_space_1)) +\
                     list(cartesian_product(hyperparameters_space_2)) +\
                     list(cartesian_product(hyperparameters_space_3))

    path = '/home/pminervi/workspace/inferbeddings/logs/ucl_countries_adv_v1/'

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
                completed = '### MICRO (test filtered)' in content

        if not completed:
            command_line = '{} >> {} 2>&1'.format(to_cmd(cfg, _path=args.path), logfile)
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
#$ -l h_vmem=2G,tmem=2G
#$ -l h_rt=1:00:00

""".format(nb_jobs)

    print(header)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        print('test $SGE_TASK_ID -eq {} && {}'.format(job_id, command_line))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
