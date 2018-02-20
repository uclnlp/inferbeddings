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
              '-f -n -m cbilstm --batch-size 32 --dropout-keep-prob 0.8 ' \
              '--representation-size 300 --optimizer adam --learning-rate 0.001 -c 100 -i uniform ' \
              '--nb-epochs 10 --has-bos --has-unk -p ' \
              '-S --restore models/snli/cbilstm_1/cbilstm_1 ' \
              '--07 {} --08 {} --09 {} --10 {} --12 {} ' \
              '-P {} --atopk {} --anc {} --anepb {} ' \
              '{}' \
              '--hard-save models/snli/cbilstm_1/acl_v2/batch_dsearch_reg_v1c/dam_1_{}'\
        .format(c['weight'], c['weight'], c['weight'], c['weight'], c['weight'],
                c['adversarial_pooling'], c['atopk'], c['anc'], c['anepb'],
                suffix, idx)
    return command


def to_logfile(c, path):
    outfile = "%s/batch_dsearch_reg_v1c.%s.log" % (path, summary(c))
    return outfile


def main(argv):
    hyperparameters_space_1 = dict(
        weight=[0.0, 0.0001, 0.001, 0.01, 0.1, 1.0], adversarial_pooling=['sum'],
        atopk=[-1], anc=[1], anepb=[1],
        af=[False], ac=[False], ar=[False]
    )
    hyperparameters_space_2 = dict(
        weight=[0.0, 0.0001, 0.001, 0.01, 0.1, 1.0], adversarial_pooling=['sum'],
        atopk=[-1], anc=[1], anepb=[1],
        af=[True], ac=[False], ar=[False]
    )
    hyperparameters_space_3 = dict(
        weight=[0.0, 0.0001, 0.001, 0.01, 0.1, 1.0], adversarial_pooling=['sum'],
        atopk=[-1], anc=[1], anepb=[1],
        af=[False], ac=[True], ar=[False]
    )
    hyperparameters_space_4 = dict(
        weight=[0.0, 0.0001, 0.001, 0.01, 0.1, 1.0], adversarial_pooling=['sum'],
        atopk=[-1], anc=[1], anepb=[1],
        af=[False], ac=[False], ar=[True]
    )

    conf_1 = list(cartesian_product(hyperparameters_space_1))
    conf_2 = list(cartesian_product(hyperparameters_space_2))
    conf_3 = list(cartesian_product(hyperparameters_space_3))
    conf_4 = list(cartesian_product(hyperparameters_space_4))

    configurations = conf_1 + conf_2 + conf_3 + conf_4

    path = './logs/nli/acl_v2/batch_dsearch_reg_v1c/'
    _path = './models/snli/cbilstm_1/acl_v2/batch_dsearch_reg_v1c/'

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
