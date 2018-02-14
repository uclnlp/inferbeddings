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
    command = 'PYTHONPATH=. python3 ./bin/nli-dsearch-reg-cli.py ' \
              '-f -n -m esim1 --batch-size 32 --dropout-keep-prob 0.5 ' \
              '--representation-size 300 --optimizer adam --learning-rate 0.0004 -c 100 -i uniform ' \
              '--nb-epochs 10 --has-bos --has-unk -p ' \
              '-S --restore models/snli/esim_1/esim_1 ' \
              '--07 {} --08 {} --09 {} --10 {} --12 {} ' \
              '-P {} ' \
              '--hard-save models/snli/esim_1/acl/batch_dsearch_reg_v12e/esim_1_{}'\
        .format(c['weight'], c['weight'], c['weight'], c['weight'], c['weight'],
                c['adversarial_pooling'], idx)
    return command


def to_logfile(c, path):
    outfile = "%s/batch_dsearch_reg_v12e.%s.log" % (path, summary(c))
    return outfile


def main(argv):
    hyperparameters_space_1 = dict(
        weight=[0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        adversarial_pooling=['sum', 'mean', 'max']
    )

    configurations = list(cartesian_product(hyperparameters_space_1))

    path = './logs/nli/acl/batch_dsearch_reg_v12e/'
    _path = './models/snli/esim_1/acl/batch_dsearch_reg_v12e/'

    # Check that we are on the UCLCS cluster first
    if os.path.exists('/home/pasquale/'):
        # If the folder that will contain logs does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(_path):
            os.makedirs(_path)

    configurations = list(configurations)

    command_lines = []
    for idx, cfg in enumerate(configurations):
        logfile = to_logfile(cfg, path)

        completed = False
        if os.path.isfile(logfile):
            with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = 'Epoch 9/1' in content

        if not completed:
            command_line = '{} > {} 2>&1'.format(to_cmd(cfg, idx), logfile)
            command_lines += [command_line]

    nb_jobs = len(command_lines)

    for job_id, command_line in enumerate(command_lines, 1):
        print('echo "Starting job {}" && {}'.format(job_id, command_line))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
