# -*- coding: utf-8 -*-

import itertools
import os
import os.path

import sys
import argparse
import logging
import numpy as np
from time import time

#experiments on sampled_small data

EXPERIMENTS = ['impl']

confs = ['0.0']

versions = ['0', '1', '2', '3', '4']  #different random seeds


EXPERIMENTS = ['{}_c{}_v{}'.format(exp, conf, version) for exp in EXPERIMENTS for conf in confs for version in versions]
#EXPERIMENTS = ['exp_symm', 'exp_impl', 'exp_impl_inv', 'exp_impl_conj', 'exp_trans_single', 'exp_trans_diff']

#USER = '/home/pminervi/workspace/'
USER = '/users/tdmeeste/workspace/'

def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs])

def to_cmd_closed_form(c, _path=None):
    if _path is None:
        _path = USER + 'inferbeddings/'
    command = 'python3 {}/bin/kbp-cli.py --auc' \
              ' --train {}/data/synth/sampled_small/{}_train.tsv' \
              ' --valid {}/data/synth/sampled_small/{}_valid.tsv' \
              ' --test {}/data/synth/sampled_small/{}_test.tsv' \
              ' --nb-epochs {}' \
              ' --lr {}' \
              ' --nb-batches {}' \
              ' --model {}' \
              ' --similarity {}' \
              ' --loss hinge' \
              ' --margin 1' \
              ' --embedding-size {}' \
              ' --subsample-size {}' \
              ' --discriminator-epochs {}' \
              ''.format(_path, _path, c['tag'], _path, c['tag'], _path, c['tag'],
                        c['epochs'], c['lr'], c['batches'],
                        c['model'],
                        c['similarity'],
                        c['embedding_size'],
                        c['subsample_size'],
                        c['disc_epochs'])
    if c['entity_space'] == 'unit-cube':
        command += ' --unit-cube'
    if c['adv_closed_form'] and c['use_clauses']:
        command += ' --adv-closed-form'

    if c['use_clauses']:
        command += ' --clauses {}/data/synth/sampled_small/{}_clauses.pl'.format(_path, c['tag'])

    #weight for closed form loss
    command += ' --adv-weight-simple {}'.format(c['clause_weight'])
    command += ' --adv-weight-simple-inverse {}'.format(c['clause_weight'])
    return command


def to_cmd_iterative(c, _path=None):
    if _path is None:
        _path = USER + 'inferbeddings/'
    command = 'python3 {}/bin/kbp-cli.py --auc' \
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
              ' --adv-pooling max' \
              ''.format(_path, _path, c['tag'], _path, c['tag'], _path, c['tag'], _path, c['tag'],
                        c['epochs'], c['lr'], c['batches'],
                        c['model'],
                        c['similarity'],
                        c['embedding_size'],
                        c['subsample_size'],
                        c['adv_lr'], c['adv_epochs'],
                        c['disc_epochs'], c['adv_weight'], c['adv_batch_size'])
    if c['adv_init_ground']:
       command += ' --adv-init-ground'
    if c['entity_space'] == 'unit-cube':
        command += ' --unit-cube'

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

    """
    closed form experiments
    """
    time_file = 'synth_paper_timing_results.txt'
    time_fID = open(time_file, 'w')
    time_fID.close()

    hyperparameters_space_distmult_complex = dict(
        tag=EXPERIMENTS,
        epochs=[100],
        lr=[0.1],
        batches=[10],
        model=['ComplEx'],
        similarity=['dot'],
        entity_space=['unit-cube'],
        embedding_size=[20],
        subsample_size=[1],
        disc_epochs=[10],
        adv_closed_form=[True],
        use_clauses=[True],
        clause_weight=[1.]#[1.e-2, 1.e-1, 1., 1.e1, 1.e2, 1.e3]
    )
    configurations_distmult_complex = cartesian_product(hyperparameters_space_distmult_complex)
    configurations = list(configurations_distmult_complex)

    t00= time()
    for job_id, cfg in enumerate(configurations):

        t0 = time()
        os.system(to_cmd_closed_form(cfg, _path=args.path))
        duration = time() - t0
        with open(time_file,'a') as time_fID:
            time_fID.write('{}\t{}\n'.format(duration, to_logfile(cfg, '')))

    with open(time_file, 'a') as time_fID:
        time_fID.write('Closed form experiments: on average %.1fs per configuration (%d runs)\n'%((time()-t00)/len(configurations),len(configurations)))

    """
    iterative experiments
    """
    hyperparameters_space_distmult_complex = dict(
        tag=EXPERIMENTS,
        epochs=[100],
        lr=[0.1],
        batches=[10],
        model=['ComplEx'],
        similarity=['dot'],
        embedding_size=[20],
        subsample_size=[1],
        adv_lr=[0.1],
        adv_epochs=[10],
        disc_epochs=[10],
        adv_weight=[1],
        adv_batch_size=[100],
        adv_init_ground=[True],
        entity_space=['unit-cube']
    )
    configurations_distmult_complex = cartesian_product(hyperparameters_space_distmult_complex)
    configurations = list(configurations_distmult_complex)

    t00 = time()
    for job_id, cfg in enumerate(configurations):

        t0 = time()
        os.system(to_cmd_iterative(cfg, _path=args.path))
        duration = time() - t0
        with open(time_file,'a') as time_fID:
            time_fID.write('{}\t{}\n'.format(duration, to_logfile(cfg, '')))

    with open(time_file, 'a') as time_fID:
        time_fID.write('Iterative experiments: on average %.1fs per configuration (%d runs)\n'%((time()-t00)/len(configurations),len(configurations)))



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
