# -*- coding: utf-8 -*-

import abc
import sys
import tensorflow as tf


class TOperators(metaclass=abc.ABCMeta):
    """
    Abstract class implementing a generic interface for a set of T-operators (T-norm, T-conorm and negation function).
    The general properties of a set of T-operators are described in [1].

    [1] Gupta, M. M. et al. - Theory of T-norms and fuzzy inference methods - Fuzzy Sets and Systems, Vol. 40, 1991, 431-450.
    """
    @abc.abstractmethod
    def norm(self, x, y):
        """
        Abstract interface for a generic T-norm.

        :param x: (N,) Tensor containing the first terms of the t-norm.
        :param y: (N,) Tensor containing the second terms of the t-norm.
        :return: (N,) Tensor containing the resulting t-norm values.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def conorm(self, x, y):
        """
        Abstract interface for a generic T-conorm.

        :param x: (N,) Tensor containing the first terms of the t-norm.
        :param y: (N,) Tensor containing the second terms of the t-norm.
        :return: (N,) Tensor containing the resulting t-norm values.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def negation(self, x):
        """
        Abstract interface for a generic negation function.

        :param x: (N,) Tensor containing the first terms of the t-norm.
        :return: (N,) Tensor containing the resulting t-norm values.
        """
        raise NotImplementedError


class ZadehTOperators(TOperators):
    """
    Zadeh's T-operators [1, 2].

    [1] Zadeh, L. A. - Outline of a new approach to the analysis of complex systems and decision processes - IEEE Trans. Systems Man Cybernet. 3 (1973) 28-44.
    [2] Gupta, M. M. et al. - Theory of T-norms and fuzzy inference methods - Fuzzy Sets and Systems, Vol. 40, 1991, 431-450.
    """
    def norm(self, x, y):
        """
        .. math:: \top(x, y) = min(x, y)

        :param x: (N,) Tensor containing the first arguments of the t-norm.
        :param y: (N,) Tensor containing the second arguments of the t-norm.
        :return: (N,) Tensor containing the resulting t-norm values.
        """
        return tf.minimum(x, y)

    def conorm(self, x, y):
        """
        .. math:: \top*(x, y) = max(x, y)

        :param x: (N,) Tensor containing the first arguments of the t-conorm.
        :param y: (N,) Tensor containing the second arguments of the t-conorm.
        :return: (N,) Tensor containing the resulting t-conorm values.
        """
        return tf.maximum(x, y)

    def negation(self, x):
        """
        .. math:: \neg(x) = 1 - x

        :param x: (N,) Tensor containing the arguments of the functional negation.
        :return: (N,) Tensor containing the resulting negated values.
        """
        return 1 - x


class ProbabilisticTOperators(TOperators):
    """
    Probabilistic T-operators [1, 2].

    [1] Weber, S. - A general concept of fuzzy connectives, negations and implications based on t-norms and t-conorms - Fuzzy Sets and Systems 11 (1983) 115-134.
    [2] Gupta, M. M. et al. - Theory of T-norms and fuzzy inference methods - Fuzzy Sets and Systems, Vol. 40, 1991, 431-450.
    """
    def norm(self, x, y):
        """
        .. math:: \top(x, y) = x * y

        :param x: (N,) Tensor containing the first arguments of the t-norm.
        :param y: (N,) Tensor containing the second arguments of the t-norm.
        :return: (N,) Tensor containing the resulting t-norm values.
        """
        return x * y

    def conorm(self, x, y):
        """
        .. math:: \top*(x, y) = x + y - xy

        :param x: (N,) Tensor containing the first arguments of the t-conorm.
        :param y: (N,) Tensor containing the second arguments of the t-conorm.
        :return: (N,) Tensor containing the resulting t-conorm values.
        """
        return x + y - x * y

    def negation(self, x):
        """
        .. math:: \neg(x) = 1 - x

        :param x: (N,) Tensor containing the arguments of the functional negation.
        :return: (N,) Tensor containing the resulting negated values.
        """
        return 1 - x


class LukasiewiczTOperators(TOperators):
    """
    Łukasiewicz T-operators [1, 2]

    [1] Giles, R. - Lukasiewicz logic and fuzzy set theory - Internat. J. Man-Machine Stud. 8 (1976) 313-327.
    [2] Gupta, M. M. et al. - Theory of T-norms and fuzzy inference methods - Fuzzy Sets and Systems, Vol. 40, 1991, 431-450.
    """
    def norm(self, x, y):
        """
        .. math:: \top(x, y) = max(0, x + y - 1)

        :param x: (N,) Tensor containing the first arguments of the t-norm.
        :param y: (N,) Tensor containing the second arguments of the t-norm.
        :return: (N,) Tensor containing the resulting t-norm values.
        """
        return tf.maximum(0, x + y - 1)

    def conorm(self, x, y):
        """
        .. math:: \top*(x, y) = min(1, x + y)

        :param x: (N,) Tensor containing the first arguments of the t-conorm.
        :param y: (N,) Tensor containing the second arguments of the t-conorm.
        :return: (N,) Tensor containing the resulting t-conorm values.
        """
        return tf.minimum(1, x + y)

    def negation(self, x):
        """
        .. math:: \neg(x) = 1 - x

        :param x: (N,) Tensor containing the arguments of the functional negation.
        :return: (N,) Tensor containing the resulting negated values.
        """
        return 1 - x


class GuptaTOperators(TOperators):
    """
    Gupta T-operators [1]

    [1] Gupta, M. M. et al. - Theory of T-norms and fuzzy inference methods - Fuzzy Sets and Systems, Vol. 40, 1991, 431-450.
    """
    def norm(self, x, y):
        """
        .. math:: \top(x, y) = xy / (x + y - xy)

        :param x: (N,) Tensor containing the first arguments of the t-norm.
        :param y: (N,) Tensor containing the second arguments of the t-norm.
        :return: (N,) Tensor containing the resulting t-norm values.
        """
        return (x * y) / (x + y - x * y)

    def conorm(self, x, y):
        """
        .. math:: \top*(x, y) = (x + y - 2xy) / (1 - x * y)

        :param x: (N,) Tensor containing the first arguments of the t-conorm.
        :param y: (N,) Tensor containing the second arguments of the t-conorm.
        :return: (N,) Tensor containing the resulting t-conorm values.
        """
        return (x + y - 2 * x * y) / (1 - x * y)

    def negation(self, x):
        """
        .. math:: \neg(x) = 1 - x

        :param x: (N,) Tensor containing the arguments of the functional negation.
        :return: (N,) Tensor containing the resulting negated values.
        """
        return 1 - x


class HamacherTOperators(TOperators):
    """
    Hamacher T-operators [1, 2]

    [1] Weber, S. - A general concept of fuzzy connectives, negations and implications based on t-norms and t-conorms - Fuzzy Sets and Systems 11 (1983) 115-134.
    [2] Gupta, M. M. et al. - Theory of T-norms and fuzzy inference methods - Fuzzy Sets and Systems, Vol. 40, 1991, 431-450.
    """
    def __init__(self, lamda=1.0):
        super().__init__()
        self.lamda = lamda

    def norm(self, x, y):
        """
        .. math:: \top(x, y) = \lambda xy / (1 - (1 - \lambda) (x + y - xy))

        :param x: (N,) Tensor containing the first arguments of the t-norm.
        :param y: (N,) Tensor containing the second arguments of the t-norm.
        :return: (N,) Tensor containing the resulting t-norm values.
        """
        return (self.lamda * x * y) / (1 - (1 - self.lamda) * (x + y - x * y))

    def conorm(self, x, y):
        """
        .. math:: \top*(x, y) = (\lambda (x + y) + xy (1 - 2 \lambda)) / (\lambda + xy (1 - \lambda))

        :param x: (N,) Tensor containing the first arguments of the t-conorm.
        :param y: (N,) Tensor containing the second arguments of the t-conorm.
        :return: (N,) Tensor containing the resulting t-conorm values.
        """
        return (self.lamda * (x + y) + x * y * (1 - 2 * self.lamda)) / (self.lamda + x * y (1 - self.lamda))

    def negation(self, x):
        """
        .. math:: \neg(x) = 1 - x

        :param x: (N,) Tensor containing the arguments of the functional negation.
        :return: (N,) Tensor containing the resulting negated values.
        """
        return 1 - x


zadeh = Zadeh = ZadehTOperators
probabilistic = Probabilistic = ProbabilisticTOperators
lukasiewicz = Lukasiewicz = LukasiewiczTOperators
gupta = Gupta = GuptaTOperators
hamacher = Hamacher = HamacherTOperators


def get_function(function_name):
    this_module = sys.modules[__name__]
    if not hasattr(this_module, function_name):
        raise ValueError('Unknown operators: {}'.format(function_name))
    return getattr(this_module, function_name)
