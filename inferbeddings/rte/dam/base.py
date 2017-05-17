# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np
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

    @abstractmethod
    def _transform_aggregate(self, v1_v2, reuse=False):
        raise NotImplementedError

    def __init__(self, optimizer, vocab_size, embedding_size=300,
                 clip_value=100.0, l2_lambda=None, trainable_embeddings=True,
                 use_masking=False, prepend_null_token=False):
        self.num_classes = 3

        self.sentence1 = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence1')
        self.sentence2 = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence2')

        self.sentence1_size = tf.placeholder(dtype=tf.int32, shape=[None], name='sent1_size')
        self.sentence2_size = tf.placeholder(dtype=tf.int32, shape=[None], name='sent2_size')

        sentence1_size = self.sentence1_size
        sentence2_size = self.sentence2_size

        self.label = tf.placeholder(dtype=tf.int32, shape=[None], name='label')

        self.embeddings = tf.get_variable('embeddings', shape=[vocab_size, embedding_size],
                                          initializer=tf.random_normal_initializer(0.0, 1.0),
                                          trainable=trainable_embeddings)
        self.transformed_embeddings = self._transform_embeddings(self.embeddings)

        # [batch, time_steps, embedding_size]
        self.embedded1 = tf.nn.embedding_lookup(self.transformed_embeddings, self.sentence1)
        # [batch, time_steps, embedding_size]
        self.embedded2 = tf.nn.embedding_lookup(self.transformed_embeddings, self.sentence2)

        self.null_token_embedding = None
        if prepend_null_token:
            self.null_token_embedding = tf.get_variable('null_embedding', shape=[1, embedding_size],
                                                        initializer=tf.random_normal_initializer(0.0, 1.0),
                                                        trainable=True)
            transformed_null_token_embedding = self._transform_embeddings(self.null_token_embedding, reuse=True)
            batch_size = tf.shape(self.embedded1)[0]
            tiled_null_token_embedding = tf.tile(input=tf.expand_dims(transformed_null_token_embedding, axis=0),
                                                 multiples=[batch_size, 1, 1])
            self.embedded1 = tf.concat(values=[tiled_null_token_embedding, self.embedded1], axis=1)
            self.embedded2 = tf.concat(values=[tiled_null_token_embedding, self.embedded2], axis=1)

            sentence1_size += 1
            sentence2_size += 1

        logger.info('Building the Attend graph ..')
        self.raw_attentions = None
        self.attention_sentence1 = self.attention_sentence2 = None

        # tensors with shape (batch_size, time_steps, num_units)
        self.alpha, self.beta = self.attend(self.embedded1, self.embedded2,
                                            sequence1_lengths=sentence1_size,
                                            sequence2_lengths=sentence2_size,
                                            use_masking=use_masking)

        logger.info('Building the Compare graph ..')
        # tensor with shape (batch_size, time_steps, num_units)
        self.v1 = self.compare(self.embedded1, self.beta)
        # tensor with shape (batch_size, time_steps, num_units)
        self.v2 = self.compare(self.embedded2, self.alpha, reuse=True)

        logger.info('Building the Aggregate graph ..')
        self.logits = self.aggregate(self.v1, self.v2, self.num_classes,
                                     v1_lengths=sentence1_size,
                                     v2_lengths=sentence2_size,
                                     use_masking=use_masking)

        self.predictions = tf.argmax(self.logits, axis=1, name='predictions')

        self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label)
        self.loss = tf.reduce_mean(self.losses)

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

    def attend(self, sequence1, sequence2,
               sequence1_lengths=None, sequence2_lengths=None, use_masking=False):
        """
        Attend phase.
        
        :param sequence1: tensor with shape (batch_size, time_steps, num_units)
        :param sequence2: tensor with shape (batch_size, time_steps, num_units)
        :param sequence1_lengths: time_steps in sequence1
        :param sequence2_lengths: time_steps in sequence2
        :param use_masking: use masking
        :return: two tensors with shape (batch_size, time_steps, num_units)
        """
        with tf.variable_scope('attend') as _:
            # tensor with shape (batch_size, time_steps, num_units)
            transformed_sequence1 = self._transform_attend(sequence1)

            # tensor with shape (batch_size, time_steps, num_units)
            transformed_sequence2 = self._transform_attend(sequence2, True)

            # tensor with shape (batch_size, time_steps, time_steps)
            self.raw_attentions = tf.matmul(transformed_sequence1, tf.transpose(transformed_sequence2, [0, 2, 1]))

            masked_raw_attentions = self.raw_attentions
            if use_masking:
                masked_raw_attentions = util.mask_3d(sequences=masked_raw_attentions,
                                                     sequence_lengths=sequence2_lengths,
                                                     mask_value=- np.inf, dimension=2)
            self.attention_sentence1 = util.attention_softmax3d(masked_raw_attentions)

            # tensor with shape (batch_size, time_steps, time_steps)
            attention_transposed = tf.transpose(self.raw_attentions, [0, 2, 1])
            masked_attention_transposed = attention_transposed
            if use_masking:
                masked_attention_transposed = util.mask_3d(sequences=masked_attention_transposed,
                                                           sequence_lengths=sequence1_lengths,
                                                           mask_value=- np.inf, dimension=2)
            self.attention_sentence2 = util.attention_softmax3d(masked_attention_transposed)

            # tensors with shape (batch_size, time_steps, num_units)
            alpha = tf.matmul(self.attention_sentence2, sequence1, name='alpha')
            beta = tf.matmul(self.attention_sentence1, sequence2, name='beta')
            return alpha, beta

    def compare(self, sentence, soft_alignment, reuse=False):
        """
        Compare phase.
        
        :param sentence: tensor with shape (batch_size, time_steps, num_units)
        :param soft_alignment: tensor with shape (batch_size, time_steps, num_units)
        :param reuse: reuse variables
        :return: tensor with shape (batch_size, time_steps, num_units)
        """
        # tensor with shape (batch, time_steps, num_units)
        sentence_and_alignment = tf.concat(axis=2, values=[sentence, soft_alignment])
        transformed_sentence_and_alignment = self._transform_compare(sentence_and_alignment, reuse=reuse)
        return transformed_sentence_and_alignment

    def aggregate(self, v1, v2, num_classes,
                  v1_lengths=None, v2_lengths=None, use_masking=False):
        """
        Aggregate phase.
        
        :param v1: tensor with shape (batch_size, time_steps, num_units)
        :param v2: tensor with shape (batch_size, time_steps, num_units)
        :param num_classes: number of output units
        :param v1_lengths: time_steps in v1
        :param v2_lengths: time_steps in v2
        :param use_masking: use masking
        :return: 
        """
        with tf.variable_scope('aggregate') as _:
            if use_masking:
                v1 = util.mask_3d(sequences=v1, sequence_lengths=v1_lengths, mask_value=0, dimension=1)
                v2 = util.mask_3d(sequences=v2, sequence_lengths=v1_lengths, mask_value=0, dimension=1)
            v1_sum, v2_sum = tf.reduce_sum(v1, [1]), tf.reduce_sum(v2, [1])
            v1_v2 = tf.concat(axis=1, values=[v1_sum, v2_sum])
            transformed_v1_v2 = self._transform_aggregate(v1_v2)
            logits = tf.contrib.layers.fully_connected(inputs=transformed_v1_v2,
                                                       num_outputs=num_classes,
                                                       weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                                       biases_initializer=tf.zeros_initializer(),
                                                       activation_fn=None)
        return logits
