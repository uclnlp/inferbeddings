# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import tensorflow as tf

from inferbeddings.models import similarities

import logging

logger = logging.getLogger(__name__)


class EquivalentPredicateRegularizer(metaclass=ABCMeta):
    def __init__(self, x1, x2, is_inverse=False, similarity_name='l2_sqr', *args, **kwargs):
        self.x1, self.x2 = x1, x2
        self.is_inverse = is_inverse
        self.similarity_name = similarity_name

    @abstractmethod
    def __call__(self):
        pass


class TransEEquivalentPredicateRegularizer(EquivalentPredicateRegularizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def inverse(self, x):
        return - x

    def __call__(self):
        similarity = similarities.get_function(self.similarity_name)
        loss = - similarity(self.x1, self.inverse(self.x2) if self.is_inverse else self.x2, axis=-1)
        return loss


class DistMultEquivalentPredicateRegularizer(EquivalentPredicateRegularizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def inverse(self, x):
        return x

    def __call__(self):
        similarity = similarities.get_function(self.similarity_name)
        loss = - similarity(self.x1, self.inverse(self.x2) if self.is_inverse else self.x2, axis=-1)
        return loss


class ComplExEquivalentPredicateRegularizer(EquivalentPredicateRegularizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def inverse(self, x):
        # TensorFlow does not allow for tf.split([..], axis=-1)
        x_re, x_im = tf.split(value=x, num_or_size_splits=2, axis=len(x.get_shape()) - 1)
        return tf.concat(values=[x_re, - x_im], axis=-1)

    def __call__(self):
        similarity = similarities.get_function(self.similarity_name)
        loss = - similarity(self.x1, self.inverse(self.x2) if self.is_inverse else self.x2, axis=-1)
        return loss
