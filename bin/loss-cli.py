#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sample usage: $ ./bin/loss-cli.py -m models/wn18/wn18_v1.pkl -c data/wn18/clauses/clauses_0.9.pl
"""

import argparse
import pickle

import tensorflow as tf
from inferbeddings.parse import parse_clause

import os
import sys
import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))




def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Rule-based Ground Loss', formatter_class=formatter)

    argparser.add_argument('--parameters', '-p', action='store', type=str, required=True, help='Model parameters')
    argparser.add_argument('--model', '-m', action='store', type=str, required=True, help='TensorFlow saved session')
    argparser.add_argument('--clauses', '-c', action='store', type=str, required=True, help='Horn clauses')

    args = argparser.parse_args(argv)

    parameters_path = args.parameters
    assert parameters_path is not None

    model_path = args.model
    assert model_path is not None

    clauses_path = args.clauses
    assert clauses_path is not None

    with open(parameters_path, 'rb') as f:
        parameters = pickle.load(f)

    with open(clauses_path, 'r') as f:
        clauses = [parse_clause(line.strip()) for line in f.readlines()]

    entity_to_index = parameters['entity_to_index']
    predicate_to_index = parameters['predicate_to_index']

    entities, predicates = set(entity_to_index.keys()), set(predicate_to_index.keys())
    nb_entities, nb_predicates = len(entities), len(predicates)

    entity_embedding_size = parameters['entities'].shape[1]
    predicate_embedding_size = parameters['predicates'].shape[1]

    entity_embedding_layer = tf.get_variable('entities', shape=[nb_entities + 1, entity_embedding_size], initializer=tf.contrib.layers.xavier_initializer())
    predicate_embedding_layer = tf.get_variable('predicates', shape=[nb_predicates + 1, predicate_embedding_size], initializer=tf.contrib.layers.xavier_initializer())

    saver = tf.train.Saver()

    with tf.Session() as session:
        saver.restore(session, model_path)

        logger.info('Model restored')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
