# -*- coding: utf-8 -*-

import abc
import numpy as np


class AIndexGenerator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, n_samples, indices):
        while False:
            yield None


class UniformIndexGenerator(AIndexGenerator):
    def __init__(self, random_state=None):
        self.random_state = random_state if random_state is not None else np.random.RandomState(0)

    def __call__(self, n_samples, indices):
        if isinstance(indices, list):
            indices = np.array(indices)

        rand_ints = self.random_state.random_integers(0, indices.size - 1, n_samples)
        return indices[rand_ints]


class GlorotIndexGenerator(AIndexGenerator):
    def __init__(self, random_state=None):
        self.random_state = random_state if random_state is not None else np.random.RandomState(0)

    def __call__(self, n_samples, indices):
        if isinstance(indices, list):
            indices = np.array(indices)

        shuffled_indices = indices[self.random_state.permutation(len(indices))]
        rand_ints = shuffled_indices[np.arange(n_samples) % len(shuffled_indices)]
        return rand_ints
