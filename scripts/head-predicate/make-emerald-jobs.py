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
              ' --train {}/data/wn18/wordnet-mlj12-train.txt' \
              ' --valid {}/data/wn18/wordnet-mlj12-valid.txt' \
              ' --test {}/data/wn18/wordnet-mlj12-test.txt' \
              ' --clauses {}/data/wn18/clauses/clauses_0.9.pl' \
              ' --nb-epochs {}' \
              ' --lr {}' \
              ' --nb-batches {}' \
              ' --model {}' \
              ' --similarity {}' \
              ' --margin {}' \
              ' --embedding-size {}' \
              ' --head-subsample-size {}' \
              ' --loss {}' \
              ' --adv-lr {} --adv-init-ground --adversary-epochs {}' \
              ' --discriminator-epochs {} --adv-weight {} --adv-batch-size {}' \
              ''.format(_path, _path, _path, _path, _path,
                        c['epochs'], c['lr'], c['batches'],
                        c['model'], c['similarity'],
                        c['margin'], c['embedding_size'],
                        c['subsample_size'],
                        c['loss'],
                        c['adv_lr'], c['adv_epochs'],
                        c['disc_epochs'], c['adv_weight'], c['adv_batch_size'])
    return command


def to_logfile(c, path):
    outfile = "%s/ucl_wn18_adv_predicate_xshot_v1.%s.log" % (path, summary(c))
    return outfile


def main(argv):
    pass

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
