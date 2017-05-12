# -*- coding: utf-8 -*-

import tensorflow as tf

from inferbeddings.rte.dam import AbstractDecomposableAttentionModel


class FeedForwardDAM(AbstractDecomposableAttentionModel):
    def __init__(self, representation_size=200, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.representation_size = representation_size

    def _transform_embeddings(self, embeddings, reuse=False):
        with tf.variable_scope('transform_embeddings', reuse=reuse) as scope:
            projection = tf.contrib.layers.fully_connected(inputs=embeddings, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.1),
                                                           biases_initializer=tf.zeros_initializer(),
                                                           activation_fn=None, scope=scope)
        return projection

    def _transform_attend(self, sequence, reuse=False):
        with tf.variable_scope('transform_attend', reuse=reuse) as scope:
            projection = tf.contrib.layers.fully_connected(inputs=sequence, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.1),
                                                           biases_initializer=tf.zeros_initializer(),
                                                           activation_fn=tf.nn.relu, scope=scope)
        return projection

    def _transform_compare(self, sequence, reuse=False):
        with tf.variable_scope('transform_compare', reuse=reuse) as scope:
            projection = tf.contrib.layers.fully_connected(inputs=sequence, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.1),
                                                           biases_initializer=tf.zeros_initializer(),
                                                           activation_fn=tf.nn.relu, scope=scope)
        return projection
