# -*- coding: utf-8 -*-

import tensorflow as tf

import inferbeddings.activations as activations
from inferbeddings.rte.dam import AbstractDecomposableAttentionModel


class DAMP(AbstractDecomposableAttentionModel):
    def __init__(self, representation_size=200, dropout_keep_prob=1.0, *args, **kwargs):
        self.representation_size = representation_size
        self.dropout_keep_prob = dropout_keep_prob
        super().__init__(*args, **kwargs)

    def _transform_embeddings(self, embeddings, reuse=False):
        with tf.variable_scope('transform_embeddings', reuse=reuse) as _:
            projection = tf.contrib.layers.fully_connected(inputs=embeddings, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                                           activation_fn=None)
        return projection

    def _transform_attend(self, sequence, reuse=False):
        with tf.variable_scope('transform_attend', reuse=reuse) as _:
            projection = tf.nn.dropout(sequence, keep_prob=self.dropout_keep_prob)
            projection = tf.contrib.layers.fully_connected(inputs=projection, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                                           biases_initializer=tf.zeros_initializer())
            projection = activations.prelu(projection, name='1')
            projection = tf.nn.dropout(projection, keep_prob=self.dropout_keep_prob)
            projection = tf.contrib.layers.fully_connected(inputs=projection, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                                           biases_initializer=tf.zeros_initializer())
            projection = activations.prelu(projection, name='2')
        return projection

    def _transform_compare(self, sequence, reuse=False):
        with tf.variable_scope('transform_compare', reuse=reuse) as _:
            projection = tf.nn.dropout(sequence, keep_prob=self.dropout_keep_prob)
            projection = tf.contrib.layers.fully_connected(inputs=projection, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                                           biases_initializer=tf.zeros_initializer())
            projection = activations.prelu(projection, name='1')
            projection = tf.nn.dropout(projection, keep_prob=self.dropout_keep_prob)
            projection = tf.contrib.layers.fully_connected(inputs=projection, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                                           biases_initializer=tf.zeros_initializer())
            projection = activations.prelu(projection, name='2')
        return projection

    def _transform_aggregate(self, v1_v2, reuse=False):
        with tf.variable_scope('transform_aggregate', reuse=reuse) as _:
            projection = tf.nn.dropout(v1_v2, keep_prob=self.dropout_keep_prob)
            projection = tf.contrib.layers.fully_connected(inputs=projection, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                                           biases_initializer=tf.zeros_initializer())
            projection = activations.prelu(projection, name='1')
            projection = tf.nn.dropout(projection, keep_prob=self.dropout_keep_prob)
            projection = tf.contrib.layers.fully_connected(inputs=projection, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                                           biases_initializer=tf.zeros_initializer())
            projection = activations.prelu(projection, name='2')
        return projection
