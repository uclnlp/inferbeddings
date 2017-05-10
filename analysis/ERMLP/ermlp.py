#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

import numpy as np
import tensorflow as tf

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))

entity_embedding_size = 200
predicate_embedding_size = 200

seed = 0

nb_epochs = 1000


def read_triples(path):
    triples = []
    with open(path, 'rt') as f:
        for line in f.readlines():
            s, p, o = line.split()
            triples += [(s.strip(), p.strip(), o.strip())]
    return triples


class ERMLP:
    def __init__(self, subject_embeddings=None, predicate_embeddings=None, object_embeddings=None, hidden_size=1, f=tf.tanh):
        self.subject_embeddings, self.object_embeddings = subject_embeddings, object_embeddings
        self.predicate_embeddings = predicate_embeddings
        self.f = f

        subject_emb_size = self.subject_embeddings.get_shape()[-1].value
        predicate_emb_size = self.predicate_embeddings.get_shape()[-1].value
        object_emb_size = self.object_embeddings.get_shape()[-1].value

        input_size = subject_emb_size + object_emb_size + predicate_emb_size

        self.C = tf.get_variable('C', shape=[input_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer())
        self.w = tf.get_variable('w', shape=[hidden_size, 1], initializer=tf.contrib.layers.xavier_initializer())

    def __call__(self):
        e_ijk = tf.concat(values=[self.subject_embeddings, self.object_embeddings, self.predicate_embeddings], axis=1)
        h_ijk = tf.matmul(e_ijk, self.C)
        f_ijk = tf.squeeze(tf.matmul(self.f(h_ijk), self.w), axis=1)

        return f_ijk


def main(argv):
    train_triples = read_triples('wn18.triples.train')
    valid_triples = read_triples('wn18.triples.valid')
    test_triples = read_triples('wn18.triples.test')

    all_triples = train_triples + valid_triples + test_triples

    entity_set = set([s for (s, p, o) in all_triples] + [o for (s, p, o) in all_triples])
    predicate_set = set([p for (s, p, o) in all_triples])

    nb_entities, nb_predicates = len(entity_set), len(predicate_set)

    entity_to_idx = {entity: idx for idx, entity in enumerate(sorted(entity_set))}
    predicate_to_idx = {predicate: idx for idx, predicate in enumerate(sorted(predicate_set))}

    entity_embedding_layer = tf.get_variable('entities', shape=[nb_entities, entity_embedding_size],
                                             initializer=tf.contrib.layers.xavier_initializer())

    predicate_embedding_layer = tf.get_variable('predicates', shape=[nb_predicates, predicate_embedding_size],
                                                initializer=tf.contrib.layers.xavier_initializer())



    with tf.Session() as session:
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
