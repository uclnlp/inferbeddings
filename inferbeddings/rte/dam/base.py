# -*- coding: utf-8 -*-

import tensorflow as tf

import logging

logger = logging.getLogger(__name__)


class MultiFFN:
    def __init__(self, optimizer, num_units, num_classes, vocab_size,
                 embedding_size=100, dropout_keep_prob=1.0, clip_value=100.0, l2_lambda=None):

        self.num_classes = 3

        self.sentence1 = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence1')
        self.sentence2 = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence2')

        self.sentence1_size = tf.placeholder(dtype=tf.int32, shape=[None], name='sent1_size')
        self.sentence2_size = tf.placeholder(dtype=tf.int32, shape=[None], name='sent2_size')

        self.label = tf.placeholder(dtype=tf.int32, shape=[None], name='label')

        self.embeddings = tf.get_variable('embeddings', shape=[vocab_size, embedding_size],
                                          initializer=tf.contrib.layers.xavier_initializer())

        # [batch, time_steps, embedding_size]
        self.embedded1 = tf.nn.embedding_lookup(self.embeddings, self.sentence1)
        # [batch, time_steps, embedding_size]
        self.embedded2 = tf.nn.embedding_lookup(self.embeddings, self.sentence2)

        logger.info('Building the Attend graph ..')
        # tensors with shape (batch_size, time_steps, num_units)
        self.alpha, self.beta = MultiFFN.attend(self.embedded1, self.embedded2)

        logger.info('Building the Compare graph ..')
        # tensor with shape (batch_size, time_steps, num_units)
        self.v1 = MultiFFN.compare(self.embedded1, self.beta)
        # tensor with shape (batch_size, time_steps, num_units)
        self.v2 = MultiFFN.compare(self.embedded2, self.alpha)

        logger.info('Building the Aggregate graph ..')
        self.logits = MultiFFN.aggreate(self.v1, self.v2, self.num_classes)

    @staticmethod
    def attention_softmax3d(values):
        """
        Performs a softmax over the attention values.
        :param values: tensor with shape (batch_size, time_steps, time_steps)
        :return: tensor with shape (batch_size, time_steps, time_steps)
        """
        original_shape = tf.shape(values)
        # tensor with shape (batch_size * time_steps, time_steps)
        reshaped_values = tf.reshape(tensor=values, shape=[-1, original_shape[2]])
        # tensor with shape (batch_size * time_steps, time_steps)
        softmax_reshaped_values = tf.nn.softmax(reshaped_values)
        # tensor with shape (batch_size, time_steps, time_steps)
        return tf.reshape(softmax_reshaped_values, original_shape)

    @staticmethod
    def attend(sequence1, sequence2):
        """
        Attend phase.
        
        :param sequence1: tensor with shape (batch_size, time_steps, num_units)
        :param sequence2: tensor with shape (batch_size, time_steps, num_units)
        :return: two tensors with shape (batch_size, time_steps, num_units)
        """
        transformed_sequence1 = sequence1
        transformed_sequence2 = sequence2

        # tensor with shape (batch_size, num_units, time_steps)
        transposed_sequence2 = tf.transpose(transformed_sequence2, [0, 2, 1])
        # tensor with shape (batch_size, time_steps, time_steps)
        raw_attentions = tf.matmul(transformed_sequence1, transposed_sequence2)
        attention_sentence1 = MultiFFN.attention_softmax3d(raw_attentions)

        # tensor with shape (batch_size, time_steps, time_steps)
        attention_transposed = tf.transpose(raw_attentions, [0, 2, 1])
        attention_sentence2 = MultiFFN.attention_softmax3d(attention_transposed)

        # tensors with shape (batch_size, time_steps, num_units)
        alpha = tf.matmul(attention_sentence2, sequence1, name='alpha')
        beta = tf.matmul(attention_sentence1, sequence2, name='beta')
        return alpha, beta

    @staticmethod
    def compare(sentence, soft_alignment):
        """
        Compare phase.
        
        :param sentence: tensor with shape (batch_size, time_steps, num_units)
        :param soft_alignment: tensor with shape (batch_size, time_steps, num_units)
        :return: tensor with shape (batch_size, time_steps, num_units)
        """
        # tensor with shape (batch, time_steps, num_units)
        sentence_and_alignment = tf.concat(axis=2, values=[sentence, soft_alignment])
        transformed_sentence_and_alignment = sentence_and_alignment
        return transformed_sentence_and_alignment

    @staticmethod
    def aggreate(v1, v2, num_classes):
        """
        Aggregate phase.
        
        :param v1: tensor with shape (batch_size, time_steps, num_units)
        :param v2: tensor with shape (batch_size, time_steps, num_units)
        :param num_classes: number of output units
        :return: 
        """
        v1_sum, v2_sum = tf.reduce_sum(v1, [1]), tf.reduce_sum(v2, [1])
        v1_v2 = tf.concat(axis=1, values=[v1_sum, v2_sum])
        logits = tf.contrib.layers.fully_connected(inputs=v1_v2, num_outputs=num_classes,
                                                   weights_initializer=tf.random_normal_initializer(0.0, 0.1),
                                                   biases_initializer=tf.zeros_initializer(), activation_fn=None)
        return logits
