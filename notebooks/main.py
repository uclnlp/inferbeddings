#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import tensorflow as tf
from inferbeddings.parse import parse_clause
from base import Inferbeddings

import logging


def main(argv):
    triples = [
        ('a', 'p', 'b'),
        ('c', 'p', 'd')
    ]
    clauses = [parse_clause('p(X, Y) :- p(Y, X)')]

    inferbeddings = Inferbeddings(triples,
                                  entity_embedding_size=10,
                                  predicate_embedding_size=10,
                                  model_name='DistMult',
                                  similarity_name='dot')

    init_op = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init_op)

        inferbeddings.init_adversary(clauses=clauses)
        for epoch in range(1, 11):
            inferbeddings.train_adversary(session=session)
            inferbeddings.train_discriminator(session=session)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
