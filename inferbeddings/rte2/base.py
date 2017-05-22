# -*- coding: utf-8 -*-

import abc

import logging

logger = logging.getLogger(__name__)


class BaseRTEModel(metaclass=abc.ABCMeta):
    def __init__(self, sequence1, sequence1_length, sequence2, sequence2_length,
                 nb_classes=3, reuse=False):
        """
        Abstract class inherited by all RTE models.

        :param sequence1: (batch_size, time_steps, embedding_size) float32 Tensor
        :param sequence1_length: (batch_size) int Tensor
        :param sequence2: (batch_size, time_steps, embedding_size) float32 Tensor
        :param sequence2_length: (batch_size) int Tensor
        :param nb_classes: number of classes
        :param reuse: reuse variables
        """
        self.nb_classes = nb_classes

        self.sequence1 = sequence1
        self.sequence1_length = sequence1_length

        self.sequence2 = sequence2
        self.sequence2_length = sequence2_length

        self.reuse = reuse

    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError
