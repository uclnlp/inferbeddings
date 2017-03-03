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
        return - similarity(self.x1, self.inverse(self.x2) if self.is_inverse else self.x2, axis=-1)


class DistMultEquivalentPredicateRegularizer(EquivalentPredicateRegularizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def inverse(self, x):
        return x

    def __call__(self):
        similarity = similarities.get_function(self.similarity_name)
        return - similarity(self.x1, self.inverse(self.x2) if self.is_inverse else self.x2, axis=-1)


class ComplExEquivalentPredicateRegularizer(EquivalentPredicateRegularizer):
    def __init__(self, embedding_size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_size = embedding_size
        assert self.embedding_size is not None

    def inverse(self, x):
        x_re, x_im = x[:self.embedding_size], x[self.embedding_size:]
        return tf.concat(values=[x_re, x_im], axis=0)

    def __call__(self):
        similarity = similarities.get_function(self.similarity_name)
        return - similarity(self.x1, self.inverse(self.x2) if self.is_inverse else self.x2, axis=-1)
