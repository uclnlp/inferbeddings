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

        # TODO - Relation indices are not changed. Corrupting relations should be an option.
        negative_steps = steps

        # Entity (subject and object) indices are corrupted for generating two new sets of walks
        entities_corr = np.copy(entities)
        entities_corr[:, 1 if self.corrupt_objects else 0] = self.index_generator(nb_samples, self.candidate_indices)

        return negative_steps, entities_corr


class InverseCorruptor(ACorruptor):
    def __call__(self, steps, entities):
        """
        """

        # TODO - Relation indices are not changed. Corrupting relations should be an option.
        negative_steps = steps

        # Entity (subject and object) indices are corrupted for generating two new sets of walks
        entities_corr = np.copy(entities)
        entities_corr[:, 0] = entities[:, 1]
        entities_corr[:, 1] = entities[:, 0]

        return negative_steps, entities_corr
