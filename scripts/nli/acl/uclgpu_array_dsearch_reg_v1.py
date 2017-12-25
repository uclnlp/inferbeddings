#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import os.path

import sys
import logging


def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs])


def to_cmd(c, idx):
    command = 'PYTHONPATH=. xpy ./bin/nli-dsearch-reg-cli.py -f -n -m ff-dam --batch-size 32 --dropout-keep-prob 0.8 ' \
              '--representation-size 200 --optimizer adagrad --learning-rate 0.05 -c 100 -i uniform ' \
              '--nb-epochs 10 --has-bos --has-unk -p ' \
              '-S --restore models/snli/dam_1/dam_1 --{} {} -P {} ' \
              '-E data/snli/generated/snli_1.0_contradictions_*.gz ' \
              '--hard-save models/snli/dam_1/acl/uclgpu_dsearch_reg_v1/dam_1_{}'\
        .format(c['rule_id'], c['weight'], c['adversarial_pooling'], idx)
    return command


def to_logfile(c, path):
    outfile = "%s/uclgpu_dsearch_reg_v1.%s.log" % (path, summary(c))
    return outfile


def main(argv):
    hyperparameters_space_1 = dict(
        rule_id=['04', '05', '06'],
        weight=[0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        adversarial_pooling=['sum', 'mean', 'max']
    )

    configurations = list(cartesian_product(hyperparameters_space_1))

    path = '/home/pminervi/workspace/inferbeddings/logs/nli/acl/uclgpu_dsearch_reg_v1/'

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
                completed = 'Epoch 8/1' in content

        if not completed:
            command_line = '{} > {} 2>&1'.format(to_cmd(cfg, idx), logfile)
            command_lines += [command_line]

    nb_jobs = len(command_lines)

    header = """#!/bin/bash

#$ -cwd
#$ -S /bin/bash
#$ -o /dev/null
#$ -e /dev/null
#$ -t 1-{}
#$ -l tmem=12G
#$ -l h_rt=12:00:00
#$ -l gpu=1,gpu_pascal=1
#$ -P gpu
#$ -l tscratch=1G

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64/:$LD_LIBRARY_PATH"

cd /home/pminervi/workspace/inferbeddings/
mkdir -p models/snli/dam_1/acl/uclgpu_dsearch_reg_v1/

""".format(nb_jobs)

    print(header)

    for job_id, command_line in enumerate(command_lines, 1):
        print('sleep 10 && test $SGE_TASK_ID -eq {} && {}'.format(job_id, command_line))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
