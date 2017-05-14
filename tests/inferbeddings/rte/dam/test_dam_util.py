# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import inferbeddings.rte.dam.util as util
import logging

import pytest

logger = logging.getLogger(__name__)


def test_mask_3d():
    batch_size = 1
    time_steps, embedding_size = 5, 4
    sequence_lengths = [2]

    tensor = tf.get_variable('M', shape=[batch_size, time_steps, embedding_size],
                             initializer=tf.random_normal_initializer(0.0, 1.0))
    masked_tensor = util.mask_3d(sequences=tensor, sequence_lengths=sequence_lengths,
                                 mask_value=- np.inf, dimension=1)

    init_op = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init_op)
        tensor_value, masked_tensor_value = session.run([tensor, masked_tensor])

        np.testing.assert_allclose(tensor_value[:, :2, :], masked_tensor_value[:, :2, :])


def test_attention_softmax3d():
    batch_size = 1
    time_steps = 16

    tensor = tf.get_variable('embeddings', shape=[batch_size, time_steps, time_steps],
                             initializer=tf.random_normal_initializer(0.0, 1.0))
    attention = util.attention_softmax3d(tensor)

    init_op = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init_op)
        tensor_value, attention_value = session.run([tensor, attention])

        assert tensor_value.shape == (batch_size, time_steps, time_steps)
        assert attention_value.shape == (batch_size, time_steps, time_steps)

        np.testing.assert_allclose(attention_value[0].sum(axis=1), np.ones(shape=time_steps), rtol=1e-4)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    pytest.main([__file__])
