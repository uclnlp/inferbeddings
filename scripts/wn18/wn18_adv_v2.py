#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import os.path


def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs])


def to_cmd(c):
    command = 'python3 ./bin/adv-cli.py' \
              ' --train data/wn18/wordnet-mlj12-train.txt' \
              ' --valid data/wn18/wordnet-mlj12-valid.txt' \
              ' --test data/wn18/wordnet-mlj12-test.txt' \
              ' --nb-epochs {}' \
              ' --lr {}' \
              ' --nb-batches {}' \
              ' --model {}' \
              ' --similarity {}' \
              ' --margin {}' \
              ' --entity-embedding-size {}' \
              ' --clauses data/wn18/clauses/clauses_0.9.pl' \
              ' --adv-lr {} --adv-nb-epochs {} --adv-weight {} ' \
              ' --adv-restart'.format(c['epochs'], c['lr'], c['batches'], c['model'], c['similarity'],
                                      c['margin'], c['embedding_size'], c['adv_lr'], c['adv_nb_epochs'], c['adv_weight'])
    return command


def to_logfile(c, path):
    outfile = "%s/wn18_adv_v2.%s.log" % (path, summary(c))
    return outfile


hyperparameters_space = dict(
    epochs=[100],
    optimizer=['adagrad'],
    lr=[.1],
    batches=[10],
    model=['ComplEx'],
    similarity=['dot'],
    margin=[1, 2, 5, 10],
    embedding_size=[10, 20, 50, 100, 150, 200, 300],

    adv_lr=[.01, .1, 1],
    adv_nb_epochs=[1, 5, 10, 25, 50, 100],
    adv_weight=[0, 1, 10, 100, 1000, 10000]
)

configurations = cartesian_product(hyperparameters_space)

path = 'logs/wn18_adv_v2/'

for cfg in configurations:
    logfile = to_logfile(cfg, path)

    completed = False
    if os.path.isfile(logfile):
        with open(logfile, 'r') as f:
            content = f.read()
            completed = '### MICRO (test filtered)' in content

    if not completed:
        line = '{} >> {} 2>&1'.format(to_cmd(cfg), logfile)
        print(line)
