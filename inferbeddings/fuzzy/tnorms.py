# -*- coding: utf-8 -*-

import tensorflow as tf

import sys


def minimum(a, b):
    """
    Minimum t-norm, also called the Gödel t-norm, is the standard semantics for conjunction in Gödel fuzzy logic.
    It occurs in most t-norm based fuzzy logics as the standard semantics for weak conjunction.

    .. math:: \top_min(a, b) = min(a, b)

    :param a: (N,) Tensor containing the first terms of the t-norm.
    :param b: (N,) Tensor containing the second terms of the t-norm.
    :return: (N,) Tensor containing the resulting t-norm values.
    """
    return tf.minimum(a, b)


def product(a, b):
    """
    Product t-norm corresponds to the ordinary product of real numbers.
    The product t-norm is the standard semantics for strong conjunction in product fuzzy logic.
    It is a strict Archimedean t-norm.

    .. math:: \top_prod(a, b) = a * b

    :param a: (N,) Tensor containing the first terms of the t-norm.
    :param b: (N,) Tensor containing the second terms of the t-norm.
    :return: (N,) Tensor containing the resulting t-norm values.
    """
    return a * b


def lukasiewicz(a, b):
    """
    Łukasiewicz t-norm: the name comes from the fact that the t-norm is the standard semantics for
    strong conjunction in Łukasiewicz fuzzy logic.
    It is a nilpotent Archimedean t-norm, pointwise smaller than the product t-norm.

    .. math:: \top_Luk(a, b) = max(0, a + b - 1)

    :param a: (N,) Tensor containing the first terms of the t-norm.
    :param b: (N,) Tensor containing the second terms of the t-norm.
    :return: (N,) Tensor containing the resulting t-norm values.
    """
    return tf.nn.relu(a + b - 1)


def nilpotent_minimum(a, b):
    """
    Nilpotent minimum t-norm is a standard example of a t-norm which is left-continuous, but not continuous.
    Despite its name, the nilpotent minimum is not a nilpotent t-norm.

    .. math:: \top_D(a, b) = if-then-else(a + b > 1, min(a, b), 0)

    :param a: (N,) Tensor containing the first terms of the t-norm.
    :param b: (N,) Tensor containing the second terms of the t-norm.
    :return: (N,) Tensor containing the resulting t-norm values.
    """
    return tf.cond(a + b > 1, tf.minimum(a, b), 0)


def hamacher_product(a, b):
    """
    Hamacher product is a strict Archimedean t-norm, and an important representative of the parametric classes of
    Hamacher t-norms and Schweizer–Sklar t-norms.

    .. math:: \top_H0(a, b) = if-then-else(a == b == 0, 0, (a * b) / (a + b - a * b))

    :param a: (N,) Tensor containing the first terms of the t-norm.
    :param b: (N,) Tensor containing the second terms of the t-norm.
    :return: (N,) Tensor containing the resulting t-norm values.
    """
    return tf.cond(a == b == 0, 0, (a * b) / (a + b - a * b))


def get_function(function_name):
    this_module = sys.modules[__name__]
    if not hasattr(this_module, function_name):
        raise ValueError('Unknown t-norm: {}'.format(function_name))
    return getattr(this_module, function_name)
