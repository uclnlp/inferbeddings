# -*- coding: utf-8 -*-

import tensorflow as tf


def least_squares_objective(output, target, add_bias=True):
    y = output
    if add_bias:
        bias = tf.Variable([0.0])
        y = output + bias
    loss = tf.reduce_sum(tf.square(y - target))
    return y, loss


def logistic_objective(output, target, add_bias=True):
    y = output
    if add_bias:
        bias = tf.Variable([0.0])
        y = output + bias
    squashed_y = tf.clip_by_value(tf.sigmoid(y), 0.001, 0.999)
    loss = -tf.reduce_sum(target*tf.log(squashed_y) + (1-target)*tf.log(1-squashed_y))
    return squashed_y, loss


def ranking_margin_objective(output, margin=1.0):
    y_pairs = tf.reshape(output, [-1, 2])
    pos_scores, neg_scores = tf.split(1, 2, y_pairs)
    hinge_losses = tf.nn.relu(margin - pos_scores + neg_scores)
    total_hinge_loss = tf.reduce_sum(hinge_losses)
    return output, total_hinge_loss
