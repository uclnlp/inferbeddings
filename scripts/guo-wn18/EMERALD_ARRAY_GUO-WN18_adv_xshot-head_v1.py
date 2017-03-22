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
    command = 'python3 {}/bin/adv-cli.py' \
              ' --train {}/data/guo-emnlp16/wn18/wn18.triples.train' \
              ' --valid {}/data/guo-emnlp16/wn18/wn18.triples.valid' \
              ' --test {}/data/guo-emnlp16/wn18/wn18.triples.test' \
              ' --clauses {}/data/guo-emnlp16/wn18/clauses/wn18-clauses.pl' \
              ' --nb-epochs {}' \
              ' --lr 0.1' \
              ' --nb-batches 10' \
              ' --model {}' \
              ' --similarity {}' \
              ' --margin {}' \
              ' --embedding-size {}' \
              ' --head-subsample-size {}' \
              ' --loss {}' \
              ' --adv-lr {} --adv-init-ground --adversary-epochs {}' \
              ' --discriminator-epochs {} --adv-weight {} --adv-batch-size {} --adv-pooling {}' \
              ''.format(_path, _path, _path, _path, _path,
                        c['epochs'],
                        c['model'], c['similarity'],
                        c['margin'], c['embedding_size'],
                        c['subsample_size'],
                        c['loss'],
                        c['adv_lr'], c['adv_epochs'],
                        c['disc_epochs'], c['adv_weight'], c['adv_batch_size'], c['adv_pooling'])
    return command


def to_logfile(c, path):
    outfile = "%s/ucl_guo-wn18_adv_xshot-head_v1.%s.log" % (path, summary(c))
    return outfile


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Generating experiments for the EMERALD cluster', formatter_class=formatter)
    argparser.add_argument('--debug', '-D', action='store_true', help='Debug flag')
    argparser.add_argument('--path', '-p', action='store', type=str, default=None, help='Path')

    args = argparser.parse_args(argv)

    hyperparameters_space_distmult_complex = dict(
        epochs=[100],
        model=['DistMult', 'ComplEx'],
        similarity=['dot'],
        margin=[1],
        embedding_size=[20, 50, 100, 150, 200],
        loss=['hinge'],
        subsample_size=[1],
        adv_lr=[.1],
        adv_epochs=[0, 10],
        disc_epochs=[10],
        adv_weight=[0, 1, 100, 10000, 1000000],
        adv_batch_size=[1, 10, 100],
        # adv_pooling=['mean', 'max']
        adv_pooling=['sum', 'mean', 'max', 'logsumexp']
    )

    configurations_distmult_complex = cartesian_product(hyperparameters_space_distmult_complex)

    path = '/home/ucl/eisuc296/workspace/inferbeddings/logs/ucl_guo-wn18_adv_xshot-head_v1/'

    # Check that we are on the UCLCS cluster first
    if os.path.exists('/home/ucl/'):
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
                completed = '### MICRO (test filtered)' in content

        if not completed:
            command_line = '{} >> {} 2>&1'.format(to_cmd(cfg, _path=args.path), logfile)
            command_lines |= {command_line}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines)
    nb_jobs = len(sorted_command_lines)

    header = """#BSUB -o /dev/null
#BSUB -e /dev/null
#BSUB -J "myarray[1-{}]"
#BSUB -W 4:00

alias python3="LD_LIBRARY_PATH='${HOME}/utils/libc6_2.17/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}' '${HOME}/utils/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so' $(command -v python3)"

export CUDA_VISIBLE_DEVICES=`~/bin/lugpu.sh`

""".format(nb_jobs)

    print(header)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        print('test $LSB_JOBINDEX -eq {} && {}'.format(job_id, command_line))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
