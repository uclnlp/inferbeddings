# -*- coding: utf-8 -*-

import tensorflow as tf


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
