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
        _path = '/home/ucl/eisuc296/workspace/inferbeddings/'
    command = 'python3 {}/bin/rte-cli.py --glove ~/data/glove/glove.840B.300d.txt' \
              ' --train {}/data/snli/snli_1.0_train.jsonl.gz' \
              ' --valid {}/data/snli/snli_1.0_dev.jsonl.gz' \
              ' --test {}data/snli/snli_1.0_test.jsonl.gz' \
              ' --embedding-size {}' \
              ' --batch-size {}' \
              ' --num-units {}' \
              ' --nb-epochs {}' \
              ' --dropout-keep-prob {}' \
              ' --learning-rate {}' \
              ''.format(_path, _path, _path, _path,
                        c['embedding_size'],
                        c['batch_size'],
                        c['num_units'],
                        c['nb_epochs'],
                        c['dropout_keep_prob'],
                        c['learning_rate'])
    return command


def to_logfile(c, path):
    outfile = "%s/emerald_snli_cbilstm_v2.%s.log" % (path, summary(c))
    return outfile


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Generating experiments for the EMERALD cluster', formatter_class=formatter)
    argparser.add_argument('--debug', '-D', action='store_true', help='Debug flag')
    argparser.add_argument('--path', '-p', action='store', type=str, default=None, help='Path')

    args = argparser.parse_args(argv)

    hyperparameters_space = dict(
        embedding_size=[300],
        batch_size=[32, 256, 1024],
        num_units=[100, 200, 300],
        nb_epochs=[100],
        dropout_keep_prob=[0.6, 0.8, 1.0],
        learning_rate=[0.0001, 0.0005, 0.001]
    )

    configurations_distmult_complex = cartesian_product(hyperparameters_space)

    path = '/home/ucl/eisuc296/workspace/inferbeddings/logs/rte/snli/emerald_snli_cbilstm_v2/'

    # Check that we are on the UCLCS cluster first
    if os.path.exists('/home/ucl/eisuc296/'):
        # If the folder that will contain logs does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)

    configurations = list(configurations_distmult_complex)

    command_lines = set()
    for cfg in configurations:
        logfile = to_logfile(cfg, path)

        completed = False
        if os.path.isfile(logfile):
            with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = 'Training finished' in content

        if not completed:
            command_line = '{} > {} 2>&1'.format(to_cmd(cfg, _path=args.path), logfile)
            command_lines |= {command_line}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines)
    nb_jobs = len(sorted_command_lines)

    header = """#BSUB -o /dev/null
#BSUB -e /dev/null
#BSUB -J "snli[1-""" + str(nb_jobs) + """]"
#BSUB -W 12:00
#BSUB -n 1
#BSUB -R "span[ptile=1]"

alias python3="LD_LIBRARY_PATH='${HOME}/utils/libc6_2.17/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}' '${HOME}/utils/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so' $(command -v python3)"

export CUDA_VISIBLE_DEVICES=`~/bin/lugpu.sh`

"""

    print(header)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        print('test $LSB_JOBINDEX -eq {} && {}'.format(job_id, command_line))
        print('sleep 1')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
