#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os.path


def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def to_str(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join(['{}={}'.format(k, v) for (k, v) in kvs])


def to_cmd(c):
    command = './bin/adv-cli.py' \
              ' --train data/hyper/lexical_entailment/baroni2012/facts_lex_train.tsv ' \
              '--valid data/hyper/lexical_entailment/baroni2012/facts_lex_val.tsv ' \
              '--test data/hyper/lexical_entailment/baroni2012/facts_lex_test.tsv ' \
              '--clauses data/hyper/lexical_entailment/transitivity.pl ' \
              '--model Concat2 ' \
              '--adversary-epochs {adv_epochs} ' \
              '--ent_embeddings ~/projects/jtr/jtr/data/GloVe/glove.6B.50d.txt ' \
              '--entity-embedding-size 50 ' \
              '--auc ' \
              '--adv-weight {weight} ' \
              '--adv-batch-size {adv_batch} ' \
              '--adv-ground-samples 100 ' \
              '--adv-ground-tol 0.1 ' \
              '--nb-epochs {epochs} ' \
              '--discriminator-epochs {disc_epochs} ' \
              '--subsample-prob {subsample_prob} ' \
              '--seed {seed} ' \
              '--adv-aggregate {aggregate} ' \
              '--adv-builder {builder} ' \
              '--noise-sample-dim 100 ' \
              '--project-adv-vars ' \
              '--adv-lr 0.01'.format(**c)
    return command


def to_logfile(cfg, path):
    outfile = '{}/hyper.{}.log'.format(path, to_str(cfg))
    return outfile


hyper_parameters_space = dict(
    epochs=[3],
    weight=[0.0, 10.0],
    adv_epochs=[100],
    adv_batch=[500],
    disc_epochs=[50],
    subsample_prob=[0.01],
    seed=range(0,10),
    aggregate=['max'],
    builder=['point-mass'],
)

configurations = list(cartesian_product(hyper_parameters_space))

path = '/Users/riedel/projects/inferbeddings/logs/hyper'

for cfg in configurations:
    logfile = to_logfile(cfg, path)

    completed = False
    if os.path.isfile(logfile):
        with open(logfile, 'r') as f:
            content = f.read()
            completed = 'AUC-ROC' in content

    if not completed:
        line = '{} >> {} 2>&1'.format(to_cmd(cfg), logfile)
        print(line)
