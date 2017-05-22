# -*- coding: utf-8 -*-

from abc import abstractmethod

import numpy as np
import tensorflow as tf

from inferbeddings.rte2 import BaseRTEModel

import logging

logger = logging.getLogger(__name__)


class BaseDecomposableAttentionModel(BaseRTEModel):
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

    def __init__(self, use_masking=False, prepend_null_token=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        batch1_size = tf.shape(self.sequence1)[0]
        batch2_size = tf.shape(self.sequence2)[0]

        assert batch1_size == batch2_size

        embedding1_size = tf.shape(self.sequence1)[2]
        embedding2_size = tf.shape(self.sequence2)[2]

        assert embedding1_size == embedding2_size

        # [batch_size, time_steps, embedding_size] -> [batch_size, time_steps, representation_size]
        self.transformed_sequence1 = self._transform_embeddings(self.sequence1, reuse=self.reuse)

        # [batch_size, time_steps, embedding_size] -> [batch_size, time_steps, representation_size]
        self.transformed_sequence2 = self._transform_embeddings(self.sequence2, reuse=True)

        sequence1 = self.sequence1
        sequence2 = self.sequence2

        sequence1_length = self.sequence1_length
        sequence2_length = self.sequence2_length

        self.null_token_embedding = None

        if prepend_null_token:
            # [1, 1, embedding_size]
            self.null_token_embedding = tf.get_variable('null_embedding',
                                                        shape=[1, 1, embedding1_size],
                                                        initializer=tf.random_normal_initializer(0.0, 1.0))

            # [1, 1, representation_size]
            transformed_null_token_embedding = self._transform_embeddings(self.null_token_embedding, reuse=True)

            # [batch_size, 1, representation_size]
            tiled_null_token_embedding = tf.tile(input=tf.expand_dims(transformed_null_token_embedding, axis=0),
                                                 multiples=[batch1_size, 1, 1])

            # [batch_size, time_steps + 1, representation_size]
            sequence1 = tf.concat(values=[tiled_null_token_embedding, sequence1], axis=1)

            # [batch_size, time_steps + 1, representation_size]
            sequence2 = tf.concat(values=[tiled_null_token_embedding, sequence2], axis=1)

            sequence1_length += 1
            sequence2_length += 1

        logger.info('Building the Attend graph ..')

        self.raw_attentions = None
        self.attention_sentence1 = self.attention_sentence2 = None

        # tensors with shape (batch_size, time_steps, num_units)
        self.alpha, self.beta = self.attend(self.embedded1, self.embedded2,
                                            sequence1_lengths=sentence1_size,
                                            sequence2_lengths=sentence2_size,
                                            use_masking=use_masking)

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
