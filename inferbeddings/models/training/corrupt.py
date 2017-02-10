# -*- coding: utf-8 -*-

import abc
import numpy as np


class ACorruptor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, steps, entities):
        while False:
            yield None


class SimpleCorruptor(ACorruptor):
    def __init__(self, index_generator=None, candidate_indices=None, corrupt_objects=False):
        self.index_generator = index_generator
        self.candidate_indices = candidate_indices
        self.corrupt_objects = corrupt_objects

    def __call__(self, steps, entities):
        """
        Generates sets of negative examples, by corrupting the facts (walks) provided as input.

        :param steps: [nb_samples, m] matrix containing the walk relation indices.
        :param entities: [nb_samples, 2] matrix containing subject and object indices.
        :return: ([nb_samples, 1], [nb_samples, 2]) pair containing sets of negative examples.
        """
        nb_samples = steps.shape[0]

        # Relation indices are not changed. For corrupting them, use a SimpleRelationCorruptor.
        negative_steps = steps

        # Entity (subject and object) indices are corrupted for generating two new sets of walks
        entities_corr = np.copy(entities)
        entities_corr[:, 1 if self.corrupt_objects else 0] = self.index_generator(nb_samples, self.candidate_indices)

        return negative_steps, entities_corr


class SimpleRelationCorruptor(ACorruptor):
    def __init__(self, index_generator=None, candidate_indices=None):
        self.index_generator = index_generator
        self.candidate_indices = candidate_indices

    def __call__(self, steps, entities):
        """
        Generates sets of negative examples, by corrupting the facts (walks) provided as input.

        :param steps: [nb_samples, m] matrix containing the walk relation indices.
        :param entities: [nb_samples, 2] matrix containing subject and object indices.
        :return: ([nb_samples, 1], [nb_samples, 2]) pair containing sets of negative examples.
        """
        nb_samples = steps.shape[0]

        # Corrupting the relation indices
        negative_steps = np.copy(steps)
        negative_steps[:, 0] = self.index_generator(nb_samples, self.candidate_indices)

        # We leave entities unchanged
        entities_corr = entities
        return negative_steps, entities_corr
