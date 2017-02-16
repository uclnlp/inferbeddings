# -*- coding: utf-8 -*-

import tensorflow as tf
import sys


def logistic_loss(scores, targets, *args, **kwargs):
    """
    Logistic loss as used in [1]

    [1] http://jmlr.org/proceedings/papers/v48/trouillon16.pdf

    # The following formulations are equivalent:
    >> tf.log(1 + tf.exp(- np.array([10., -5.]) * np.array([1., -1]))).eval()
    array([  4.53988992e-05,   6.71534849e-03])
    >> tf.nn.softplus(- np.array([10., -5.]) * np.array([1., -1])).eval()
    array([  4.53988992e-05,   6.71534849e-03])
    >> tf.nn.sigmoid_cross_entropy_with_logits(np.array([10., -5.]), np.array([1., 0.])).eval()
    array([  4.53988992e-05,   6.71534849e-03])

    :param scores: (N,) Tensor containing scores of examples.
    :param targets: (N,) Tensor containing {0, 1} targets of examples.
    :return: Loss value.
    """
    logistic_losses = tf.nn.sigmoid_cross_entropy_with_logits(scores, targets)
    loss = tf.reduce_sum(logistic_losses)
    return loss


def hinge_loss(scores, targets, margin=1, *args, **kwargs):
    """
    Hinge loss.
    :param scores: (N,) Tensor containing scores of examples.
    :param targets: (N,) Tensor containing {0, 1} targets of examples.
    :param margin: float representing the margin in the hinge loss relu(margin - logits * (2 * targets - 1))
    :return: Loss value.
    """
    # The following two formulations are equivalent:
    # > x.eval()
    # array([0., -1., 2., -3., 4., -5., 6.], dtype=float32)
    # > t.eval()
    # array([0., 1., 1., 1., 0., 0., 1.], dtype=float32)
    # > tf.contrib.losses.hinge_loss(logits=x, labels=t).eval()
    # array([1., 2., 0., 4., 5., 0., 0.], dtype=float32)
    # > tf.nn.relu(1 - x * (2 * t - 1)).eval()
    # array([1., 2., 0., 4., 5., 0., 0.], dtype=float32)

    # hinge_losses = tf.contrib.losses.hinge_loss(logits=scores, labels=targets)
    hinge_losses = tf.nn.relu(margin - scores * (2 * targets - 1))

    loss = tf.reduce_sum(hinge_losses)
    return loss


# Aliases
logistic = logistic_loss
hinge = hinge_loss


def get_function(function_name):
    this_module = sys.modules[__name__]
    if not hasattr(this_module, function_name):
        raise ValueError('Unknown loss function: {}'.format(function_name))
    return getattr(this_module, function_name)
