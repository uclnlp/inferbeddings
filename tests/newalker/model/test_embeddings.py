# -*- coding: utf-8 -*-

import pytest

import numpy as np
import tensorflow as tf

import inferbeddings.models.embeddings as embeddings


def test_additive_walk_embedding():
    batch_size = 5
    embedding_size = 10
    walk_length = 3

    rs = np.random.RandomState(0)
    P = rs.rand(batch_size, walk_length, embedding_size)

    vP = tf.Variable(P, name='P')
    vW = embeddings.additive_walk_embedding(vP)

    init_op = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init_op)

        swe = session.run(vW)
        assert(swe.shape[0] == batch_size)
        assert(np.allclose(swe, np.sum(P, axis=1)))


def test_additive_walk_embedding_zeros():
    batch_size = 5
    embedding_size = 10
    walk_length = 0

    rs = np.random.RandomState(0)
    P = rs.rand(batch_size, walk_length, embedding_size)

    vP = tf.Variable(P, name='P')
    vW = embeddings.additive_walk_embedding(vP)

    init_op = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init_op)

        swe = session.run(vW)
        assert(swe.shape[0] == batch_size)
        assert(np.allclose(swe, np.sum(P, axis=1)))


def test_bilinear_diagonal_walk_embedding():
    batch_size = 5
    embedding_size = 10
    walk_length = 3

    rs = np.random.RandomState(0)
    P = rs.rand(batch_size, walk_length, embedding_size)

    vP = tf.Variable(P, name='P')
    vW = embeddings.bilinear_diagonal_walk_embedding(vP)

    init_op = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init_op)

        swe = session.run(vW)
        assert(swe.shape[0] == batch_size)
        assert(np.allclose(swe, np.prod(P, axis=1)))


def test_bilinear_walk_embedding():
    batch_size = 1
    embedding_size = 25
    walk_length = 1

    rs = np.random.RandomState(0)
    P = rs.rand(batch_size, walk_length, embedding_size)

    vP = tf.Variable(P, name='P')
    vW = embeddings.bilinear_walk_embedding(vP, int(np.sqrt(embedding_size)))

    init_op = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init_op)

        swe = session.run(vW)
        assert(swe.shape[0] == batch_size)
        assert(np.allclose(swe, P.reshape(1, 5, 5)))


if __name__ == '__main__':
    pytest.main([__file__])
