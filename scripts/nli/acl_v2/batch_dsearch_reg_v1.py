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
    suffix = ' '
    suffix += ' --af ' if c['af'] is True else ''
    suffix += ' --ac ' if c['ac'] is True else ''
    suffix += ' --ar ' if c['ar'] is True else ''

    command = 'PYTHONPATH=. python3 ./bin/nli-dsearch-reg-v2-cli.py ' \
              '-f -n -m ff-dam --batch-size 32 --dropout-keep-prob 0.8 ' \
              '--representation-size 200 --optimizer adagrad --learning-rate 0.05 -c 100 -i uniform ' \
              '--nb-epochs 10 --has-bos --has-unk -p ' \
              '-S --restore models/snli/dam_1/dam_1 ' \
              '--07 {} --08 {} --09 {} --10 {} --12 {} ' \
              '-P {} {}' \
              '--hard-save models/snli/dam_1/acl_v2/batch_dsearch_reg_v1/dam_1_{}'\
        .format(c['weight'], c['weight'], c['weight'], c['weight'], c['weight'],
                c['adversarial_pooling'], suffix, idx)
    return command


def to_logfile(c, path):
    outfile = "%s/batch_dsearch_reg_v1.%s.log" % (path, summary(c))
    return outfile


def main(argv):
    hyperparameters_space_1 = dict(
        weight=[0.0, 0.0001, 0.001, 0.01, 0.1, 1.0],
        adversarial_pooling=['sum'],

        atopk=[-1],
        anc=[1, 10],
        anepb=[8, 16, 32],

        af=[True, False],
        ac=[True, False],
        ar=[True, False]
    )

    configurations = list(cartesian_product(hyperparameters_space_1))

    path = './logs/nli/acl_v2/batch_dsearch_reg_v1/'
    _path = './models/snli/dam_1/acl_v2/batch_dsearch_reg_v1/'

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

    for job_id, command_line in enumerate(command_lines, 1):
        print('echo "Starting job {}" && {}'.format(job_id, command_line))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
