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

    clause_one = parse_clause('p(X, Y) :- p(Y, X)')
    clause_two = parse_clause('p(X, Y) :- p(X, Y)')

    with tf.Session() as session:
        clauses = [
            (clause_one, 0.0),
            (clause_two, 1.0)
        ]

        inferbeddings = Inferbeddings(session=session, triples=triples, clauses=clauses)

        for epoch in range(1, 6):
            inferbeddings.train_adversary(session=session)
            inferbeddings.train_discriminator(session=session)

        clauses = [
            (clause_one, 1.0),
            (clause_two, 0.0)
        ]

        inferbeddings.init_adversary(session, clauses)

        for epoch in range(1, 6):
            inferbeddings.train_adversary(session=session)
            inferbeddings.train_discriminator(session=session)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
