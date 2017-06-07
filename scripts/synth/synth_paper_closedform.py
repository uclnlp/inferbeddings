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

#EXPERIMENTS = ['symm', 'impl', 'impl_inv', 'trans_single', 'trans_diff']
EXPERIMENTS = ['symm', 'impl', 'impl_inv']
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


"""
#some other arguments to consider:
#    argparser.add_argument('--initial-accumulator-value', action='store', type=float, default=0.1)
#    argparser.add_argument('--pairwise-loss', action='store', type=str, default='hinge_loss',
                           help='Pairwise loss function')
#    argparser.add_argument('--corrupt-relations', action='store_true',
                           help='Also corrupt the relation of each training triple for generating negative examples')
#    argparser.add_argument('--margin', '-M', action='store', type=float, default=1.0, help='Margin')
#    argparser.add_argument('--predicate-embedding-size', '-p', action='store', type=int, default=None,
                           help='Predicate embedding size')
#    argparser.add_argument('--all-one-entities', nargs='+', type=str,
                           help='Entities with all-one entity embeddings')
#    argparser.add_argument('--predicate-norm', action='store', type=float, default=None,
                           help='Norm of the predicate embeddings')
#    argparser.add_argument('--sar-weight', action='store', type=float, default=None,
                           help='Schema-Aware Regularization, regularizer weight')
#    argparser.add_argument('--sar-similarity', action='store', type=str, default='l2_sqr',
                           help='Schema-Aware Regularization, similarity measure')

#    argparser.add_argument('--adv-margin', action='store', type=float, default=0.0, help='Adversary margin')
#    argparser.add_argument('--adv-ground-samples', action='store', type=int, default=None,
                           help='Number of ground samples on which to compute the ground loss')
#    argparser.add_argument('--adv-ground-tol', '--adv-ground-tolerance', action='store', type=float, default=0.0,
                           help='Epsilon-tolerance when calculating the ground loss')

#    argparser.add_argument('--head-subsample-size', action='store', type=float, default=None,
                           help='Fraction of training facts to use during training (e.g. 0.1)')

#    argparser.add_argument('--materialize', action='store_true',
                           help='Materialize all facts using clauses and logical inference')
#    argparser.add_argument('--save', action='store', type=str, default=None,
                           help='Path for saving the serialized model')
"""



def to_cmd(c, _path=None):
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


def to_logfile(c, path):
    outfile = "%s/synth_paper_closedform.%s.log" % (path, summary(c))
    return outfile


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Generating experiments for the UCL cluster', formatter_class=formatter)
    argparser.add_argument('--debug', '-D', action='store_true', help='Debug flag')
    argparser.add_argument('--path', '-p', action='store', type=str, default=None, help='Path')

    args = argparser.parse_args(argv)

    # hyperparameters_space_transe = dict(
    #     tag=EXPERIMENTS,
    #     epochs=[100],
    #     lr=[0.1],
    #     batches=[10],
    #     model=['TransE'],
    #     similarity=['l2_sqr'],
    #     entity_space=['unit-cube', 'unit-sphere'],
    #     embedding_size=[20],
    #     subsample_size=[1],
    #     disc_epochs=[10],
    #     adv_closed_form=[True],
    #     use_clauses=[False, True],
    #     clause_weight=[1.]#[1.e-2, 1.e-1, 1., 1.e1, 1.e2, 1.e3]
    # )

    #to do adversarial training:   adv_lr=[0.1],


    hyperparameters_space_distmult_complex = dict(
        tag=EXPERIMENTS,
        epochs=[100],
        lr=[0.1],
        batches=[10],
        model=['DistMult', 'ComplEx'],
        similarity=['dot'],
        entity_space=['unit-cube', 'unit-sphere'],
        embedding_size=[20],
        subsample_size=[1],
        disc_epochs=[10],
        adv_closed_form=[True],
        use_clauses=[True],
        clause_weight=[1.]#[1.e-2, 1.e-1, 1., 1.e1, 1.e2, 1.e3]
    )

    #configurations_transe = cartesian_product(hyperparameters_space_transe)
    configurations_distmult_complex = cartesian_product(hyperparameters_space_distmult_complex)


    path = USER + 'inferbeddings/logs/synth/synth_paper_closedform'
    if not os.path.exists(path):
        os.makedirs(path)

    #configurations = list(configurations_transe) + list(configurations_distmult_complex)
    configurations = list(configurations_distmult_complex)
    np.random.shuffle(configurations)

    #prune configurations by hand, for combinations that aren't needed
    configurations_pruned = []
    for c in configurations:
        add = True
        #if c['clause_weight'] != 1. and not c['use_clauses']:
        #    add = False

        #if c['tag'] == 'trans_diff' and c['model'] in ['ComplEx', 'TransE']: #not implemented
        #    add = False

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
