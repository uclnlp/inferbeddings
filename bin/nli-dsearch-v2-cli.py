#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import os
import sys

import pickle
import socket
import copy

import numpy as np
import tensorflow as tf

from inferbeddings.io import load_glove, load_glove_words
from inferbeddings.models.training.util import make_batches

from inferbeddings.nli import util, tfutil
from inferbeddings.nli.evaluation import util as eutil
from inferbeddings.nli import ConditionalBiLSTM, FeedForwardDAM, FeedForwardDAMP, FeedForwardDAMS, ESIMv1

import inferbeddings.nli.regularizers.base as R

from inferbeddings.models.training import constraints

from inferbeddings.nli.generate.generator import Generator
from inferbeddings.nli.generate.scorer import LMScorer
from inferbeddings.nli.generate.scorer import IScorer

from inferbeddings.nli.evaluation import accuracy, stats

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    logger.info('Command line: {}'.format(' '.join(arg for arg in argv)))

    def fmt(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    def fmt(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Regularising RTE via Adversarial Sets Regularisation', formatter_class=fmt)

    argparser.add_argument('--data', '-d', action='store', type=str, default='data/snli/snli_1.0_train.jsonl.gz')

    argparser.add_argument('--model', '-m', action='store', type=str, default='ff-dam',
                           choices=['cbilstm', 'ff-dam', 'ff-damp', 'ff-dams', 'esim1'])

    argparser.add_argument('--embedding-size', action='store', type=int, default=300)
    argparser.add_argument('--representation-size', action='store', type=int, default=200)

    argparser.add_argument('--batch-size', action='store', type=int, default=32)

    argparser.add_argument('--seed', action='store', type=int, default=0)

    argparser.add_argument('--has-bos', action='store_true', default=False, help='Has <Beginning Of Sentence> token')
    argparser.add_argument('--has-eos', action='store_true', default=False, help='Has <End Of Sentence> token')
    argparser.add_argument('--has-unk', action='store_true', default=False, help='Has <Unknown Word> token')

    argparser.add_argument('--semi-sort', '-S', action='store_true')

    argparser.add_argument('--restore', action='store', type=str, default=None)

    for i in range(0, 13):
        argparser.add_argument('--rule{:02d}-weight'.format(i), '--{:02d}'.format(i),
                               action='store', type=float, default=None)

    argparser.add_argument('--adversarial-batch-size', '-B', action='store', type=int, default=32)
    argparser.add_argument('--adversarial-pooling', '-P', default='max', choices=['sum', 'max', 'mean', 'logsumexp'])

    argparser.add_argument('--report', '-r', default=10000, type=int,
                           help='Number of batches between performance reports')
    argparser.add_argument('--report-loss', default=100, type=int,
                           help='Number of batches between loss reports')

    # Parameters for adversarial training
    argparser.add_argument('--lm', action='store', type=str, default='models/lm/',
                           help='Language Model')

    # XXX: default to None (disable) - 0.01
    argparser.add_argument('--adversarial-epsilon', '--aeps',
                           action='store', type=float, default=None)
    argparser.add_argument('--adversarial-nb-corruptions', '--anc',
                           action='store', type=int, default=32)
    argparser.add_argument('--adversarial-nb-examples-per-batch', '--anepb',
                           action='store', type=int, default=4)
    # XXX: default to -1 (disable) - 4
    argparser.add_argument('--adversarial-top-k', '--atopk',
                           action='store', type=int, default=-1)

    argparser.add_argument('--adversarial-flip', '--af', action='store_true', default=False)
    argparser.add_argument('--adversarial-combine', '--ac', action='store_true', default=False)
    argparser.add_argument('--adversarial-remove', '--ar', action='store_true', default=False)

    args = argparser.parse_args(argv)

    lm_path = args.lm
    a_epsilon = args.adversarial_epsilon
    a_nb_corr = args.adversarial_nb_corruptions
    a_nb_examples_per_batch = args.adversarial_nb_examples_per_batch
    a_top_k = args.adversarial_top_k
    a_is_flip = args.adversarial_flip
    a_is_combine = args.adversarial_combine
    a_is_remove = args.adversarial_remove

    # Command line arguments
    data_path = args.data

    model_name = args.model

    embedding_size = args.embedding_size
    representation_size = args.representation_size

    batch_size = args.batch_size

    has_bos = args.has_bos
    has_eos = args.has_eos
    has_unk = args.has_unk

    restore_path = args.restore

    # Experimental RTE regularizers
    rule00_weight = args.rule00_weight
    rule01_weight = args.rule01_weight
    rule02_weight = args.rule02_weight
    rule03_weight = args.rule03_weight
    rule04_weight = args.rule04_weight
    rule05_weight = args.rule05_weight
    rule06_weight = args.rule06_weight
    rule07_weight = args.rule07_weight
    rule08_weight = args.rule08_weight
    rule09_weight = args.rule09_weight
    rule10_weight = args.rule10_weight
    rule11_weight = args.rule11_weight
    rule12_weight = args.rule12_weight

    adversarial_pooling_name = args.adversarial_pooling

    name_to_adversarial_pooling = {
        'sum': tf.reduce_sum,
        'max': tf.reduce_max,
        'mean': tf.reduce_mean,
        'logsumexp': tf.reduce_logsumexp
    }