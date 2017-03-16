#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import os.path

import sys
import argparse
import logging

#USER = '/home/pminervi/workspace/'
USER = '/users/tdmeeste/workspace/'


def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs])


def to_cmd(c, _path=None):
    if _path is None:
        _path = USER + 'inferbeddings/'
    command = 'python3 {}/bin/adv-cli.py --auc' \
              ' --train {}/data/nyt/{}' \
              ' --test {}/data/nyt/naacl2013_test.tsv' \
              ' --debug-scores {}/data/nyt/naacl2013_test_nolabel.tsv' \
              ' --clauses {}/data/nyt/naacl2013_clauses.pl' \
              ' --nb-epochs {}' \
              ' --lr {}' \
              ' --adv-lr {}' \
              ' --nb-batches {}' \
              ' --model DistMult' \
              ' --similarity dot' \
              ' --margin {}' \
              ' --embedding-size 100' \
              ' --subsample-size 1' \
              ' --pairwise-loss {}' \
              ' --adv-init-ground ' \
              ' --adversary-epochs {}' \
              ' --adv-ground-samples 100' \
              ' --adv-ground-tol 0.' \
              ' --discriminator-epochs {} --adv-weight {} --adv-batch-size {}' \
              ''.format(_path,
                        _path, c['train_file'],
                        _path,
                        _path,
                        _path,
                        c['epochs'],
                        c['lr'],
                        c['adv_lr'],
                        c['batches'],
                        c['margin'],
                        c['pairwise_loss'],
                        c['adv_epochs'],
                        c['disc_epochs'], c['adv_weight'], c['adv_batch_size'])

    return command


def to_logfile(c, path):
    outfile = "%s/nyt.%s.log" % (path, summary(c))
    return outfile


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Generating experiments for the UCL cluster', formatter_class=formatter)
    argparser.add_argument('--debug', '-D', action='store_true', help='Debug flag')
    argparser.add_argument('--path', '-p', action='store', type=str, default=None, help='Path')

    args = argparser.parse_args(argv)

    hyperparameters_space_logistic = dict(
        train_file=['naacl2013_train_FB_%.2f.tsv'%f for f in [0., 0.1, 0.2, 0.3, 0.4, 0.5]],
        epochs=[100],
        lr=[0.1],
        adv_lr=[0.1],
        batches=[10],
        margin=[1.],
        pairwise_loss=['hinge'],#, 'logistic, 'softplus'
        adv_epochs=[0, 10],
        disc_epochs=[10],
        adv_weight=[0, 1],
        adv_batch_size=[100]
    )

    configurations = cartesian_product(hyperparameters_space_logistic)
#    hyperparameters_space_hinge = {k:hyperparameters_space_logistic[k] for k in hyperparameters_space_logistic}
#    hyperparameters_space_hinge['pairwise_loss'] = ['hinge']
#    hyperparameters_space_hinge['margin'] = [1.0]


#    configurations = list(cartesian_product(hyperparameters_space_logistic)) + \
#                        list(cartesian_product(hyperparameters_space_hinge))


    #prune configurations by hand, for combinations that aren't needed
    configurations_pruned = []
    for c in configurations:
        add = True
        if c['adv_weight'] == 0 and (c['adv_epochs'] == 1 or c['adv_epochs'] == 10):
            add = False
        if add:
            configurations_pruned.append(c)

    configurations = configurations_pruned



    path = USER + 'inferbeddings/logs/nyt/nyt_xshot'
    if not os.path.exists(path):
        os.makedirs(path)

    for job_id, cfg in enumerate(configurations):
        logfile = to_logfile(cfg, path)

        completed = False
        if os.path.isfile(logfile):
            with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = 'Inverse Triple:' in content

        if not completed:
            line = '{} > {} 2>&1'.format(to_cmd(cfg, _path=args.path), logfile)
            print(line)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
