# -*- coding: utf-8 -*-

import pytest

import numpy as np
import tensorflow as tf

from inferbeddings.models.training import pairwise_losses


def test_additive_walk_embedding():
    pos_scores = tf.Variable(np.array([.9]), name='pos_scores')
    neg_scores = tf.Variable(np.array([.1]), name='neg_scores')

    init_op = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init_op)
        res = session.run(pairwise_losses.hinge_loss(pos_scores, neg_scores, margin=100.0))

        np.testing.assert_almost_equal(100 - .9 + .1, res)

if __name__ == '__main__':
    pytest.main([__file__])
