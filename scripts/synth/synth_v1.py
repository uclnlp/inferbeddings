#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import os.path

import sys
import argparse
import logging

EXPERIMENTS = ['symm',
               'impl',
               'impl_inv',
               'impl_conj',
               'trans_single',
               'trans_diff',
               'multiple'
               ]

confs = ['0.3', '0.5', '0.7']

EPOCHS = 500

EXPERIMENTS = ['{}_c{}'.format(exp, conf) for exp in EXPERIMENTS for conf in confs]
#EXPERIMENTS = ['exp_symm', 'exp_impl', 'exp_impl_inv', 'exp_impl_conj', 'exp_trans_single', 'exp_trans_diff']

#USER = '/home/pminervi/workspace/'
USER = '/users/tdmeeste/workspace/'
#USER = '~/workspace/'

"""
Notes:
- for now: validation on training data; no hyperparam tuning

"""

def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs])


def to_cmd(c, _path=None):
    if _path is None:
        _path = USER + 'inferbeddings/'
    command = 'python3 {}/bin/adv-cli.py --auc' \
              ' --train {}/data/synth/sampled/{}_train.tsv' \
              ' --valid {}/data/synth/sampled/{}_valid.tsv' \
              ' --test {}/data/synth/sampled/{}_test.tsv' \
              ' --clauses {}/data/synth/sampled/{}_clauses.pl' \
              ' --nb-epochs {}' \
              ' --lr {}' \
              ' --nb-batches {}' \
              ' --model {}' \
              ' --margin 1' \
              ' --embedding-size {}' \
              ' --subsample-size {}' \
              ' --loss {}' \
              ' --adv-lr {} --adv-init-ground --adversary-epochs {}' \
              ' --discriminator-epochs {} --adv-weight {} --adv-batch-size {}' \
              ''.format(_path, _path, c['tag'], _path, c['tag'], _path, c['tag'], _path, c['tag'],
                        c['epochs'], c['lr'], c['batches'],
                        c['model'],
                        c['embedding_size'],
                        c['subsample_size'],
                        c['loss'],
                        c['adv_lr'], c['adv_epochs'],
                        c['disc_epochs'], c['adv_weight'], c['adv_batch_size'])
    return command


def to_logfile(c, path):
    outfile = "%s/synth.%s.log" % (path, summary(c))
    return outfile


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Generating experiments for the UCL cluster', formatter_class=formatter)
    argparser.add_argument('--debug', '-D', action='store_true', help='Debug flag')
    argparser.add_argument('--path', '-p', action='store', type=str, default=None, help='Path')

    args = argparser.parse_args(argv)

    hyperparameters_space_transe = dict(
        tag=EXPERIMENTS,
        epochs=[EPOCHS],
        lr=[.1],
        batches=[5],
        model=['TransE'],
        similarity=['l1'], #['l1', 'l2'],
        embedding_size=[50], #[20, 50, 100, 150, 200],
        loss=['hinge'],
        subsample_size=[.1, .5, 1], #[.1, .2, .3, .4, .5, .6, .7, 1],
        adv_lr=[0.05, .1],
        adv_epochs=[1], #[0, 10],
        disc_epochs=[1, 10], #[10],
        adv_weight=[0, 1, 10], #[0, 1, 100], #[0, 1, 100, 10000, 1000000],
        adv_batch_size=[10]#[1, 10, 100]
    )

    hyperparameters_space_distmult_complex = dict(
        tag=EXPERIMENTS,
        epochs=[EPOCHS],
        lr=[.1],
        batches=[5],
        model=['DistMult', 'ComplEx'],
        similarity=['dot'],
        embedding_size=[50], #[20, 50, 100, 150, 200],
        loss=['hinge'],
        subsample_size=[.1, .5, 1], #[.1, .2, .3, .4, .5, .6, .7, 1],
        adv_lr=[0.05, .1],
        adv_epochs=[1], #[0, 10],
        disc_epochs=[1, 10], #[10],
        adv_weight=[0, 1, 10], #[0, 1, 100], #[0, 1, 100, 10000, 1000000],
        adv_batch_size=[10]#[1, 10, 100]
    )

    configurations_transe = cartesian_product(hyperparameters_space_transe)
    configurations_distmult_complex = cartesian_product(hyperparameters_space_distmult_complex)

    path = USER + 'inferbeddings/logs/synth/synth_v1'
    if not os.path.exists(path):
        os.makedirs(path)

    configurations = list(configurations_transe) + list(configurations_distmult_complex)

    for job_id, cfg in enumerate(configurations):
        logfile = to_logfile(cfg, path)

        completed = False
        if os.path.isfile(logfile):
            with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = '### MICRO (test filtered)' in content

        if not completed:
            line = '{} >> {} 2>&1'.format(to_cmd(cfg, _path=args.path), logfile)
            print(line)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
