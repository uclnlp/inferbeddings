# -*- coding: utf-8 -*-

import sys
import tensorflow as tf


def renorm_update(var_matrix, norm=1.0, axis=1):
    row_norms = tf.sqrt(tf.reduce_sum(tf.square(var_matrix), axis=axis))
    scaled = var_matrix * tf.expand_dims(norm / row_norms, axis=axis)
    return tf.assign(var_matrix, scaled)


def pseudoboolean_linear_update(var_matrix):
    pseudoboolean_linear = tf.minimum(1., tf.maximum(var_matrix, 0.))
    return tf.assign(var_matrix, pseudoboolean_linear)


def pseudoboolean_sigmoid_update(var_matrix):
    pseudoboolean_sigmoid = tf.nn.sigmoid(var_matrix)
    return tf.assign(var_matrix, pseudoboolean_sigmoid)


unit_sphere = renorm = renorm_update
unit_cube = pseudoboolean_linear = pseudoboolean_linear_update
pseudoboolean_sigmoid = pseudoboolean_sigmoid_update


def get_function(function_name):
    this_module = sys.modules[__name__]
    if not hasattr(this_module, function_name):
        raise ValueError('Unknown constraint: {}'.format(function_name))
    return getattr(this_module, function_name)
