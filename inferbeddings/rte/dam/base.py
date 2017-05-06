# -*- coding: utf-8 -*-

import tensorflow as tf

import logging

logger = logging.getLogger(__name__)


class MultiFFN:
    def __init__(self, optimizer, num_units, num_classes, vocab_size,
                 embedding_size=100, dropout_keep_prob=1.0, clip_value=100.0, l2_lambda=None):

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

    @staticmethod
    def attention_softmax3d(values):
        """
        Performs a softmax over the attention values.
        :param values: 3d tensor with raw values
        :return: 3d tensor, same shape as input
        """
        original_shape = tf.shape(values)
        num_units = original_shape[2]
        reshaped_values = tf.reshape(values, tf.stack([-1, num_units]))
        return tf.reshape(tf.nn.softmax(reshaped_values), original_shape)

    @staticmethod
    def attend(sentence1, representation1, sentence2, representation2,):
        raw_attentions = tf.matmul(representation1, representation2)
        attention_sentence1 = MultiFFN.attention_softmax3d(raw_attentions)

        attention_transposed = tf.transpose(raw_attentions, [0, 2, 1])
        attention_sentence2 = MultiFFN.attention_softmax3d(attention_transposed)

        alpha = tf.matmul(attention_sentence2, sentence1, name='alpha')
        beta = tf.matmul(attention_sentence1, sentence2, name='beta')
        return alpha, beta
