#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import tensorflow as tf
from base import KGEmbeddings

import logging


def main(argv):
    triples = [
        ('a', 'p', 'b'),
        ('c', 'p', 'd')
    ]

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    kge = KGEmbeddings(triples=triples, optimizer=optimizer)

    init_op = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init_op)
        kge.train(session=session, nb_epochs=10)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
