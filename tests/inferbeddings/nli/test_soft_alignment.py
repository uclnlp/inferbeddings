# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import logging

import pytest

logger = logging.getLogger(__name__)


def attention_softmax3d(values):
    """
    Performs a softmax over the attention values.
    :param values: tensor with shape (batch_size, time_steps, time_steps)
    :return: tensor with shape (batch_size, time_steps, time_steps)
    """
    original_shape = tf.shape(values)
    # tensor with shape (batch_size * time_steps, time_steps)
    reshaped_values = tf.reshape(tensor=values, shape=[-1, original_shape[2]])
    # tensor with shape (batch_size * time_steps, time_steps)
    softmax_reshaped_values = tf.nn.softmax(reshaped_values)
    # tensor with shape (batch_size, time_steps, time_steps)
    return tf.reshape(softmax_reshaped_values, original_shape)


@pytest.mark.light
def test_soft_alignment():
    batch_size = 1
    time_steps_seq1, time_steps_seq2 = 12, 16
    embedding_size = 32

    seq1 = tf.get_variable('seq1', shape=[batch_size, time_steps_seq1, embedding_size],
                           initializer=tf.random_normal_initializer(0.0, 1.0))
    seq2 = tf.get_variable('seq2', shape=[batch_size, time_steps_seq2, embedding_size],
                           initializer=tf.random_normal_initializer(0.0, 1.0))

    raw_attentions = tf.matmul(seq1, tf.transpose(seq2, [0, 2, 1]))

    init_op = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init_op)

        raw_attentions_value = session.run(raw_attentions)
        assert raw_attentions_value.shape == (batch_size, time_steps_seq1, time_steps_seq2)

        attention_seq1 = attention_softmax3d(raw_attentions)
        attention_seq2 = attention_softmax3d(tf.transpose(raw_attentions, [0, 2, 1]))

        alpha = tf.matmul(attention_seq2, seq1, name='alpha')
        beta = tf.matmul(attention_seq1, seq2, name='beta')

        assert alpha.shape == (batch_size, time_steps_seq2, embedding_size)
        assert beta.shape == (batch_size, time_steps_seq1, embedding_size)

    tf.reset_default_graph()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    pytest.main([__file__])
