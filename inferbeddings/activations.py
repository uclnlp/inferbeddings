# -*- coding: utf-8 -*-

import sys
import tensorflow as tf


def parametric_relu(x, name=None):
    alphas = tf.get_variable('{}/alpha'.format(name) if name else 'alpha',
                             x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    return tf.nn.relu(x) + alphas * (x - abs(x)) * 0.5

# Aliases
relu = tf.nn.relu
prelu = parametric_relu


def get_function(function_name):
    this_module = sys.modules[__name__]
    if not hasattr(this_module, function_name):
        raise ValueError('Unknown activation function: {}'.format(function_name))
    return getattr(this_module, function_name)
