#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os.path


def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def to_str(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join(['{}={}'.format(k, v) for (k, v) in kvs])


def to_cmd(fold_no, c):
    command = './bin/triples-cli.py -a' \
              ' --train data/nations/stratified_folds/{}/nations_train.tsv.gz' \
              ' --valid data/nations/stratified_folds/{}/nations_valid.tsv.gz' \
              ' --test data/nations/stratified_folds/{}/nations_test.tsv.gz' \
              ' --nb-epochs {}' \
              ' --lr {}' \
              ' --nb-batches {}' \
              ' --model {}' \
              ' --similarity {}' \
              ' --margin {}' \
              ' --entity-embedding-size {}'.format(fold_no, fold_no, fold_no, c['epochs'], c['lr'], c['batches'],
                                                   c['model'], c['similarity'], c['margin'], c['embedding_size'])
    return command


def to_logfile(fold_no, cfg, path):
    outfile = '{}/nations_v2.fold={}.{}.log'.format(path, fold_no, to_str(cfg))
    return outfile


hyper_parameters_space = dict(
    epochs=[1000],
    optimizer=['adagrad'],
    lr=[.001, .01, .1, 1.],
    batches=[10],
    model=['ComplEx', 'DistMult', 'TransE'],
    similarity=['dot', 'L1', 'L2'],
    margin=[1, 2, 5, 10],
    embedding_size=[10, 20, 50, 100, 150, 200],
)


configurations = list(cartesian_product(hyper_parameters_space))

path = 'logs/nations_v2/'

for fold_no in range(10):
    for cfg in configurations:
        logfile = to_logfile(fold_no, cfg, path)

        completed = False
        if os.path.isfile(logfile):
            with open(logfile, 'r') as f:
                content = f.read()
                completed = '### MICRO (test filtered)' in content

        if not completed:
            line = '{} >> {} 2>&1'.format(to_cmd(fold_no, cfg), logfile)
            print(line)
