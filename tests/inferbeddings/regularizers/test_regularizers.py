# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from inferbeddings.regularizers import TransEEquivalentPredicateRegularizer
from inferbeddings.regularizers import DistMultEquivalentPredicateRegularizer
from inferbeddings.regularizers import ComplExEquivalentPredicateRegularizer
from inferbeddings.regularizers import BilinearEquivalentPredicateRegularizer

import pytest


def l2sqr(x):
    return np.sum(np.square(x), axis=-1)


def complex_conjugate(x):
    x_re, x_im = np.split(x, indices_or_sections=2, axis=-1)
    return np.concatenate((x_re, - x_im), axis=-1)


@pytest.mark.light
def test_translations():
    rs = np.random.RandomState(0)
    pe = rs.rand(1024, 10)

    var = tf.Variable(pe, name='predicate_embedding')

    init_op = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init_op)

        loss = TransEEquivalentPredicateRegularizer(x1=var[0, :], x2=var[0, :])()
        np.testing.assert_almost_equal(session.run(loss), 0.0)

        loss = TransEEquivalentPredicateRegularizer(x1=var[0, :], x2=var[1, :])()
        np.testing.assert_almost_equal(session.run(loss), l2sqr(pe[0, :] - pe[1, :]))

        loss = TransEEquivalentPredicateRegularizer(x1=var[0:4, :], x2=var[0:4, :])()
        np.testing.assert_almost_equal(session.run(loss), [0.0] * 4)

        loss = TransEEquivalentPredicateRegularizer(x1=var[0, :], x2=var[1, :], is_inverse=True)()
        np.testing.assert_almost_equal(session.run(loss), l2sqr(pe[0, :] + pe[1, :]))

        loss = TransEEquivalentPredicateRegularizer(x1=var[0:4, :], x2=var[1:5, :])()
        np.testing.assert_almost_equal(session.run(loss), l2sqr(pe[0:4, :] - pe[1:5, :]))

        loss = TransEEquivalentPredicateRegularizer(x1=var[0:4, :], x2=var[1:5, :], is_inverse=True)()
        np.testing.assert_almost_equal(session.run(loss), l2sqr(pe[0:4, :] + pe[1:5, :]))

    tf.reset_default_graph()


@pytest.mark.light
def test_scaling():
    rs = np.random.RandomState(0)
    pe = rs.rand(1024, 10)

    var = tf.Variable(pe, name='predicate_embedding')

    init_op = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init_op)

        loss = DistMultEquivalentPredicateRegularizer(x1=var[0, :], x2=var[0, :])()
        np.testing.assert_almost_equal(session.run(loss), 0.0)

        loss = DistMultEquivalentPredicateRegularizer(x1=var[0, :], x2=var[1, :])()
        np.testing.assert_almost_equal(session.run(loss), l2sqr(pe[0, :] - pe[1, :]))

        loss = DistMultEquivalentPredicateRegularizer(x1=var[0:4, :], x2=var[0:4, :])()
        np.testing.assert_almost_equal(session.run(loss), [0.0] * 4)

        loss = DistMultEquivalentPredicateRegularizer(x1=var[0, :], x2=var[1, :], is_inverse=True)()
        np.testing.assert_almost_equal(session.run(loss), l2sqr(pe[0, :] - pe[1, :]))

        loss = DistMultEquivalentPredicateRegularizer(x1=var[0:4, :], x2=var[1:5, :])()
        np.testing.assert_almost_equal(session.run(loss), l2sqr(pe[0:4, :] - pe[1:5, :]))

        loss = DistMultEquivalentPredicateRegularizer(x1=var[0:4, :], x2=var[1:5, :], is_inverse=True)()
        np.testing.assert_almost_equal(session.run(loss), l2sqr(pe[0:4, :] - pe[1:5, :]))

    tf.reset_default_graph()


@pytest.mark.light
def test_complex():
    rs = np.random.RandomState(0)
    pe = rs.rand(1024, 10)

    var = tf.Variable(pe, name='predicate_embedding')

    init_op = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init_op)

        loss = ComplExEquivalentPredicateRegularizer(x1=var[0, :], x2=var[0, :])()
        np.testing.assert_almost_equal(session.run(loss), 0.0)

        loss = ComplExEquivalentPredicateRegularizer(x1=var[0, :], x2=var[1, :])()
        np.testing.assert_almost_equal(session.run(loss), l2sqr(pe[0, :] - pe[1, :]))

        loss = ComplExEquivalentPredicateRegularizer(x1=var[0:4, :], x2=var[0:4, :])()
        np.testing.assert_almost_equal(session.run(loss), [0.0] * 4)

        loss = ComplExEquivalentPredicateRegularizer(x1=var[0, :], x2=var[1, :], is_inverse=True)()
        np.testing.assert_almost_equal(session.run(loss), l2sqr(pe[0, :] - complex_conjugate(pe[1, :])))

        loss = ComplExEquivalentPredicateRegularizer(x1=var[0:4, :], x2=var[1:5, :])()
        np.testing.assert_almost_equal(session.run(loss), l2sqr(pe[0:4, :] - pe[1:5, :]))

        loss = ComplExEquivalentPredicateRegularizer(x1=var[0:4, :], x2=var[1:5, :], is_inverse=True)()
        np.testing.assert_almost_equal(session.run(loss), l2sqr(pe[0:4, :] - complex_conjugate(pe[1:5, :])))

    tf.reset_default_graph()


@pytest.mark.light
def test_bilinear():
    rs = np.random.RandomState(0)
    pe = rs.rand(1024, 16)

    var = tf.Variable(pe, name='predicate_embedding')

    kwargs = {'entity_embedding_size': 4}

    init_op = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init_op)

        loss = BilinearEquivalentPredicateRegularizer(x1=var[0, :], x2=var[0, :], **kwargs)()
        np.testing.assert_almost_equal(session.run(loss), 0.0)

        var3 = tf.reshape(var, [-1, 4, 4])
        s_var = tf.reshape(var3 + tf.transpose(var3, [0, 2, 1]), [-1, 16])

        loss = BilinearEquivalentPredicateRegularizer(x1=s_var[0, :], x2=s_var[0, :], is_inverse=True, **kwargs)()
        np.testing.assert_almost_equal(session.run(loss), 0.0)

        def invert(emb):
            predicate = emb
            predicate_matrix = tf.reshape(predicate, [-1, 4, 4])
            predicate_matrix_transposed = tf.transpose(predicate_matrix, [0, 2, 1])
            predicate_inverse = tf.reshape(predicate_matrix_transposed, [-1, 16])
            return predicate_inverse

        for i in range(32):
            loss = BilinearEquivalentPredicateRegularizer(x1=var[i, :], x2=invert(var[i, :]), is_inverse=True, **kwargs)()
            np.testing.assert_almost_equal(session.run(loss), 0.0)

            loss = BilinearEquivalentPredicateRegularizer(x1=var[0:32, :], x2=invert(var[0:32, :]), is_inverse=True, **kwargs)()
            np.testing.assert_almost_equal(session.run(loss), 0.0)


    tf.reset_default_graph()


if __name__ == '__main__':
    pytest.main([__file__])
