# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import tensorflow as tf
import logging

from inferbeddings.rte.dam import util

logger = logging.getLogger(__name__)


class AbstractDecomposableAttentionModel(metaclass=ABCMeta):
    @abstractmethod
    def _transform_embeddings(self, embeddings, reuse=False):
        raise NotImplementedError

    @abstractmethod
    def _transform_attend(self, sequence, reuse=False):
        raise NotImplementedError

    @abstractmethod
    def _transform_compare(self, sequence, reuse=False):
        raise NotImplementedError

    def __init__(self, optimizer, vocab_size, embedding_size=300,
                 clip_value=100.0, l2_lambda=None, trainable_embeddings=True):
        self.num_classes = 3

        self.sentence1 = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence1')
        self.sentence2 = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence2')

        self.sentence1_size = tf.placeholder(dtype=tf.int32, shape=[None], name='sent1_size')
        self.sentence2_size = tf.placeholder(dtype=tf.int32, shape=[None], name='sent2_size')

        self.label = tf.placeholder(dtype=tf.int32, shape=[None], name='label')

        self.embeddings = tf.get_variable('embeddings', shape=[vocab_size, embedding_size],
                                          initializer=tf.contrib.layers.xavier_initializer(),
                                          trainable=trainable_embeddings)
        self.transformed_embeddings = self._transform_embeddings(self.embeddings)

        # [batch, time_steps, embedding_size]
        self.embedded1 = tf.nn.embedding_lookup(self.transformed_embeddings, self.sentence1)
        # [batch, time_steps, embedding_size]
        self.embedded2 = tf.nn.embedding_lookup(self.transformed_embeddings, self.sentence2)

        logger.info('Building the Attend graph ..')
        # tensors with shape (batch_size, time_steps, num_units)
        self.alpha, self.beta = self.attend(self.embedded1, self.embedded2)

        logger.info('Building the Compare graph ..')
        # tensor with shape (batch_size, time_steps, num_units)
        self.v1 = self.compare(self.embedded1, self.beta)
        # tensor with shape (batch_size, time_steps, num_units)
        self.v2 = self.compare(self.embedded2, self.alpha)

        logger.info('Building the Aggregate graph ..')
        self.logits = self.aggreate(self.v1, self.v2, self.num_classes)

        self.predictions = tf.argmax(self.logits, axis=1, name='predictions')

        labels = tf.one_hot(self.label, self.num_classes)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=labels)

        if l2_lambda is not None:
            regularizer = l2_lambda * sum(tf.nn.l2_loss(var) for var in tf.trainable_variables()
                                          if not ('noreg' in var.name or 'bias' in var.name or 'Bias' in var.name))
            self.loss += regularizer

        if clip_value is not None:
            gradients, v = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, clip_value)
            self.training_step = optimizer.apply_gradients(zip(gradients, v))
        else:
            self.training_step = optimizer.minimize(self.loss)

    def attend(self, sequence1, sequence2):
        """
        Attend phase.
        
        :param sequence1: tensor with shape (batch_size, time_steps, num_units)
        :param sequence2: tensor with shape (batch_size, time_steps, num_units)
        :return: two tensors with shape (batch_size, time_steps, num_units)
        """
        with tf.variable_scope('attend') as scope:
            transformed_sequence1 = self._transform_attend(sequence1)
            transformed_sequence2 = self._transform_attend(sequence2, True)

            # tensor with shape (batch_size, num_units, time_steps)
            transposed_sequence2 = tf.transpose(transformed_sequence2, [0, 2, 1])
            # tensor with shape (batch_size, time_steps, time_steps)
            raw_attentions = tf.matmul(transformed_sequence1, transposed_sequence2)
            attention_sentence1 = util.attention_softmax3d(raw_attentions)

            # tensor with shape (batch_size, time_steps, time_steps)
            attention_transposed = tf.transpose(raw_attentions, [0, 2, 1])
            attention_sentence2 = util.attention_softmax3d(attention_transposed)

            # tensors with shape (batch_size, time_steps, num_units)
            alpha = tf.matmul(attention_sentence2, sequence1, name='alpha')
            beta = tf.matmul(attention_sentence1, sequence2, name='beta')
            return alpha, beta

    def compare(self, sentence, soft_alignment):
        """
        Compare phase.
        
        :param sentence: tensor with shape (batch_size, time_steps, num_units)
        :param soft_alignment: tensor with shape (batch_size, time_steps, num_units)
        :return: tensor with shape (batch_size, time_steps, num_units)
        """
        # tensor with shape (batch, time_steps, num_units)
        sentence_and_alignment = tf.concat(axis=2, values=[sentence, soft_alignment])
        transformed_sentence_and_alignment = self._transform_compare(sentence_and_alignment)
        return transformed_sentence_and_alignment

    def aggreate(self, v1, v2, num_classes):
        """
        Aggregate phase.
        
        :param v1: tensor with shape (batch_size, time_steps, num_units)
        :param v2: tensor with shape (batch_size, time_steps, num_units)
        :param num_classes: number of output units
        :return: 
        """
        with tf.variable_scope('aggregate') as scope:
            v1_sum, v2_sum = tf.reduce_sum(v1, [1]), tf.reduce_sum(v2, [1])
            v1_v2 = tf.concat(axis=1, values=[v1_sum, v2_sum])
            logits = tf.contrib.layers.fully_connected(inputs=v1_v2,
                                                       num_outputs=num_classes,
                                                       weights_initializer=tf.random_normal_initializer(0.0, 0.1),
                                                       biases_initializer=tf.zeros_initializer(),
                                                       activation_fn=None,
                                                       scope=scope)
        return logits
