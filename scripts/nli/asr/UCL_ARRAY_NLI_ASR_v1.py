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


def to_cmd(c, idx, _path=None):
    if _path is None:
        _path = '/home/pminervi/workspace/inferbeddings/'
    command = 'ssh $HOSTNAME /home/pminervi/bin/xpy-gpu -u {}/bin/nli-cli.py -f -n -m ff-dam --batch-size 32 --dropout-keep-prob 0.8 ' \
              '--representation-size 200 --optimizer adagrad --learning-rate 0.05 -c 100 -i uniform ' \
              '--nb-epochs 100 --has-bos --has-unk -p ' \
              '-S -I --restore /home/pminervi/workspace/inferbeddings/models/snli/dam_1/dam_1 -{} {} -B {} -L {} -A {} -P {} ' \
              '--hard-save /home/pminervi/workspace/inferbeddings/models/snli/dam_1/regularized/dam_1_{}'.format(_path, c['rule_id'], c['weight'],
                        c['adversarial_batch_size'], c['adversarial_sentence_length'], c['nb_adversary_epochs'],
                        c['adversarial_pooling'], idx)
    return command


def to_logfile(c, path):
    outfile = "%s/ucl_nli_asr_v1.%s.log" % (path, summary(c))
    return outfile


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Generating experiments for the UCL cluster', formatter_class=formatter)
    argparser.add_argument('--debug', '-D', action='store_true', help='Debug flag')
    argparser.add_argument('--path', '-p', action='store', type=str, default=None, help='Path')

    args = argparser.parse_args(argv)

    hyperparameters_space_1 = dict(
        rule_id=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        weight=[0.0, 0.0001, 0.01,  1.0, 100.0, 10000.0],
        adversarial_batch_size=[256],
        adversarial_sentence_length=[10],
        nb_adversary_epochs=[10],
        adversarial_pooling=['sum', 'max']  # ['sum', 'max', 'mean', 'logsumexp']
    )

    configurations = list(cartesian_product(hyperparameters_space_1))

    path = '/home/pminervi/workspace/inferbeddings/logs/nli/asr/ucl_nli_asr_v1/'

    # Check that we are on the UCLCS cluster first
    if os.path.exists('/home/pminervi/'):
        # If the folder that will contain logs does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)

    configurations = list(configurations)

    command_lines = []
    for idx, cfg in enumerate(configurations):
        logfile = to_logfile(cfg, path)

        completed = False
        if os.path.isfile(logfile):
            with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = 'Epoch 10/1' in content

        if not completed:
            command_line = '{} > {} 2>&1'.format(to_cmd(cfg, idx, _path=args.path), logfile)
            command_lines += [command_line]

    nb_jobs = len(command_lines)

    header = """#!/bin/bash

#$ -cwd
#$ -S /bin/bash
#$ -o /dev/null
#$ -e /dev/null
#$ -t 1-{}
# #$ -l h_vmem=24G,tmem=24G
#$ -l tmem=24G
#$ -l h_rt=24:00:00
#$ -P gpu
#$ -l gpu=1

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

shopt -s huponexit

cd /home/pminervi/workspace/inferbeddings/

""".format(nb_jobs)

    print(header)

    for job_id, command_line in enumerate(command_lines, 1):
        print('sleep 10 && test $SGE_TASK_ID -eq {} && {}'.format(job_id, command_line))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
