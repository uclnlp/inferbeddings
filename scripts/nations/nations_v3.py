#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os.path
import random
random.seed(1234)

NO_FOLDS = True
EPOCHS = 100

def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def to_str(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join(['{}={}'.format(k, v) for (k, v) in kvs])


def to_cmd(fold_no, c):
    command = './bin/adv-cli.py --auc ' \
              ' --train data/nations/stratified_folds/{}/nations_train.tsv.gz' \
              ' --valid data/nations/stratified_folds/{}/nations_valid.tsv.gz' \
              ' --test data/nations/stratified_folds/{}/nations_test.tsv.gz ' \
              ' --adv-init-ground '.format(fold_no, fold_no, fold_no)
    if 'clauses' in c:
        command += ' --clauses data/nations/clauses/{}'.format(c['clauses'])
    if 'epochs' in c:
        command += ' --nb-epochs {}'.format(c['epochs'])
    if 'lr' in c:
        command += ' --lr {}'.format(c['lr'])
    if 'batches' in c:
        command += ' --nb-batches {}'.format(c['batches'])
    if 'model' in c:
        command += ' --model {}'.format(c['model'])
    if 'similarity' in c:
        command += ' --similarity {}'.format(c['similarity'])
    if 'margin' in c:
        command += ' --margin {}'.format(c['margin'])
    if 'embedding_size' in c:
        command += ' --entity-embedding-size {}'.format(c['embedding_size'])
        command += ' --predicate-embedding-size {}'.format(c['embedding_size'])
    if 'adv_lr' in c:
        command += ' --adv-lr {}'.format(c['adv_lr'])
    if 'adv_epochs' in c:
        command += ' --adversary-epochs {}'.format(c['adv_epochs'])
    if 'disc_epochs' in c:
        command += ' --discriminator-epochs {}'.format(c['disc_epochs'])
    if 'adv_weight' in c:
        command += ' --adv-weight {}'.format(c['adv_weight'])
    if 'adv_batch_size' in c:
        command += ' --adv-batch-size {}'.format(c['adv_batch_size'])

    return command


def to_logfile(fold_no, cfg, path):
    outfile = '{}/nations_v3.fold={}.{}.log'.format(path, fold_no, to_str(cfg))
    return outfile


hyper_parameters_space_clauses_transe = dict(
    clauses=['clauses_conf0.950_supp50.pl'],
    epochs=[EPOCHS],
    lr=[.1,],
    batches=[10],
    model=['TransE'],
    similarity=['L1'],
    margin=[1],
    embedding_size=[50],
    adv_lr=[.1],
    adv_epochs=[1, 10],
    disc_epochs=[1, 10],
    adv_weight=[0., 1, 10],
    adv_batch_size=[1, 5, 20]
)

hyper_parameters_space_clauses_other = dict(
    clauses=['clauses_conf0.950_supp50.pl'],
    epochs=[EPOCHS],
    lr=[.1,],
    batches=[10],
    model=['DistMult', 'ComplEx'],
    similarity=['dot'],
    margin=[1],
    embedding_size=[50],
    adv_lr=[.1],
    adv_epochs=[1, 10],
    disc_epochs=[1, 10],
    adv_weight=[0., 1, 10],
    adv_batch_size=[1, 5, 20]
)

hyper_parameters_space_noclauses_transe = dict(
    epochs=[EPOCHS],
    lr=[.1],
    batches=[10],
    model=['TransE'],
    similarity=['L1'],
    margin=[1],
    embedding_size=[50],
)

hyper_parameters_space_noclauses_other = dict(
    epochs=[EPOCHS],
    lr=[.1],
    batches=[10],
    model=['DistMult', 'ComplEx'],
    similarity=['dot'],
    margin=[1],
    embedding_size=[50],
)


configurations_c = list(cartesian_product(hyper_parameters_space_clauses_transe)) \
                    + list(cartesian_product(hyper_parameters_space_clauses_other))
random.shuffle(configurations_c) #random order; to have more meaningful intermediate results

configurations_nc = list(cartesian_product(hyper_parameters_space_noclauses_transe)) \
                    + list(cartesian_product(hyper_parameters_space_noclauses_other))
random.shuffle(configurations_nc)
configurations = configurations_nc + configurations_c  #first configs without clauses


path = 'logs/nations_v3/'
if not os.path.exists(path):
    os.makedirs(path)

if NO_FOLDS:
    nfolds = 1
else:
    nfolds = 10

for fold_no in range(nfolds):
    for cfg in configurations:
        logfile = to_logfile(fold_no, cfg, path)

        completed = False
        if os.path.isfile(logfile):
            with open(logfile, 'r') as f:
                content = f.read()
                completed = '[test]' in content and 'AUC' in content

        if not completed:
            line = '{} >> {} 2>&1'.format(to_cmd(fold_no, cfg), logfile)
            print(line)
