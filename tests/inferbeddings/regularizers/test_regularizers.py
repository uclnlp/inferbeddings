# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from inferbeddings.regularizers import TransEEquivalentPredicateRegularizer,\
    DistMultEquivalentPredicateRegularizer,\
    ComplExEquivalentPredicateRegularizer

import pytest


def test_translations():
    rs = np.random.RandomState(0)
    pe = rs.rand(1024, 10)

    var = tf.Variable(pe, name='pe')

    init_op = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init_op)

        loss = TransEEquivalentPredicateRegularizer(x1=var[0, :], x2=var[0, :])()
        np.testing.assert_almost_equal(session.run(loss), 0.0)

        loss = TransEEquivalentPredicateRegularizer(x1=var[0, :], x2=var[1, :])()
        np.testing.assert_almost_equal(session.run(loss), np.sum(np.square(pe[0, :] - pe[1, :])))

if __name__ == '__main__':
    pytest.main([__file__])
