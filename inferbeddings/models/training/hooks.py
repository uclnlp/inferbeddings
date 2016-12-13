# -*- coding: utf-8 -*-

import abc
import tensorflow as tf
import time

import logging

logger = logging.getLogger(__name__)


class Hook(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, sess, epoch, model, loss):
        raise NotImplementedError


class TraceHook(Hook):
    def __init__(self, summary_writer=None):
        self.summary_writer = summary_writer

    @abc.abstractmethod
    def __tag__(self):
        raise NotImplementedError

    def update_summary(self, session, current_step, title, value):
        if self.summary_writer is not None:
            cur_summary = tf.scalar_summary(title, value)
            merged_summary_op = tf.merge_summary([cur_summary])
            summary_str = session.run(merged_summary_op)
            self.summary_writer.add_summary(summary_str, current_step)


class SpeedHook(TraceHook):
    def __init__(self, iter_interval, batch_size, summary_writer=None):
        super(SpeedHook, self).__init__(summary_writer)
        self.iter_interval = iter_interval
        self.batch_size = batch_size
        self.t0 = time.time()
        self.num_examples = iter_interval * batch_size
        self.iter = 0

    def __tag__(self):
        return 'Speed'

    def __call__(self, session, epoch, model, loss):
        self.iter += 1
        if not self.iter == 0 and self.iter % self.iter_interval == 0:
            diff = time.time() - self.t0
            speed = '{0:.2f}'.format(self.num_examples / diff)
            logger.info('Epoch {}\tIter {}\tExamples/s {}'.format(str(epoch), str(self.iter), str(speed)))
            self.update_summary(session, self.iter, self.__tag__(), float(speed))
            self.t0 = time.time()
