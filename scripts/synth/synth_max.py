#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import os.path

import sys
import argparse
import logging
import numpy as np


#experiments on sampled_small data

EXPERIMENTS = ['symm', 'impl', 'impl_inv', 'trans_single', 'trans_diff']
#EXPERIMENTS = ['impl', 'trans_diff']

confs = ['0.0']

versions = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  #different random seeds


EXPERIMENTS = ['{}_c{}_v{}'.format(exp, conf, version) for exp in EXPERIMENTS for conf in confs for version in versions]
#EXPERIMENTS = ['exp_symm', 'exp_impl', 'exp_impl_inv', 'exp_impl_conj', 'exp_trans_single', 'exp_trans_diff']

#USER = '/home/pminervi/workspace/'
USER = '/users/tdmeeste/workspace/'

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
              ' --train {}/data/synth/sampled_small/{}_train.tsv' \
              ' --valid {}/data/synth/sampled_small/{}_valid.tsv' \
              ' --test {}/data/synth/sampled_small/{}_test.tsv' \
              ' --clauses {}/data/synth/sampled_small/{}_clauses.pl' \
              ' --nb-epochs {}' \
              ' --lr {}' \
              ' --nb-batches {}' \
              ' --model {}' \
              ' --similarity {}' \
              ' --loss hinge' \
              ' --margin 1' \
              ' --embedding-size {}' \
              ' --subsample-size {}' \
              ' --adv-lr {} --adversary-epochs {}' \
              ' --discriminator-epochs {} --adv-weight {} --adv-batch-size {}' \
              ' --adv-pooling {}' \
              ''.format(_path, _path, c['tag'], _path, c['tag'], _path, c['tag'], _path, c['tag'],
                        c['epochs'], c['lr'], c['batches'],
                        c['model'],
                        c['similarity'],
                        c['embedding_size'],
                        c['subsample_size'],
                        c['adv_lr'], c['adv_epochs'],
                        c['disc_epochs'], c['adv_weight'], c['adv_batch_size'],
                        c['adv_pooling'])

    if c['adv_init_ground']:
       command += ' --adv-init-ground'

    #    ' --adv-ground-samples 100 --adv-ground-tol 0.1' \

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
        epochs=[100],
        lr=[0.1],
        batches=[10],
        model=['TransE'],
        similarity=['l1'],
        embedding_size=[20],
        subsample_size=[1],
        adv_lr=[0.1],
        adv_epochs=[0, 10],
        disc_epochs=[10],
        adv_weight=[1],
        adv_batch_size=[100],
        adv_init_ground=[True],
        adv_pooling=['max', 'mean', 'sum']
    )

    hyperparameters_space_distmult_complex = dict(
        tag=EXPERIMENTS,
        epochs=[100],
        lr=[0.1],
        batches=[10],
        model=['DistMult', 'ComplEx'],
        similarity=['dot'],
        embedding_size=[20],
        subsample_size=[1],
        adv_lr=[0.1],
        adv_epochs=[0, 10],
        disc_epochs=[10],
        adv_weight=[1],
        adv_batch_size=[100],
        adv_init_ground=[True],
        adv_pooling=['max', 'mean', 'sum']
    )

    configurations_transe = cartesian_product(hyperparameters_space_transe)
    configurations_distmult_complex = cartesian_product(hyperparameters_space_distmult_complex)


    path = USER + 'inferbeddings/logs/synth/synth_max'
    if not os.path.exists(path):
        os.makedirs(path)

    configurations = list(configurations_transe) + list(configurations_distmult_complex)
    np.random.shuffle(configurations)

    #prune configurations by hand, for combinations that aren't needed
    configurations_pruned = []
    for c in configurations:
        add = True
        if c['adv_weight'] == 0 and (c['adv_epochs'] == 1 or c['adv_epochs'] == 10):
            add = False

        if add:
            configurations_pruned.append(c)

    configurations = configurations_pruned


    for job_id, cfg in enumerate(configurations):
        logfile = to_logfile(cfg, path)

        completed = False
        if os.path.isfile(logfile):
            with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = 'AUC-PR' in content

        if not completed:
            line = '{} > {} 2>&1'.format(to_cmd(cfg, _path=args.path), logfile)
            print(line)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
