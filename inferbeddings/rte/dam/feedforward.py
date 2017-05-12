# -*- coding: utf-8 -*-

import tensorflow as tf

from inferbeddings.rte.dam import AbstractDecomposableAttentionModel


class FeedForwardDAM(AbstractDecomposableAttentionModel):
    def __init__(self, representation_size=300, dropout_keep_prob=1.0, *args, **kwargs):
        self.representation_size = representation_size
        self.dropout_keep_prob = dropout_keep_prob
        super().__init__(*args, **kwargs)

    def _transform_embeddings(self, embeddings, reuse=False):
        with tf.variable_scope('transform_embeddings', reuse=reuse) as _:
            # projection = tf.contrib.layers.fully_connected(inputs=embeddings, num_outputs=self.representation_size,
            #                                                weights_initializer=tf.random_normal_initializer(0.0, 0.01),
            #                                                biases_initializer=tf.zeros_initializer(),
            #                                                activation_fn=None)
            projection = embeddings
        return projection

    def _transform_attend(self, sequence, reuse=False):
        with tf.variable_scope('transform_attend', reuse=reuse) as _:
            projection = tf.contrib.layers.fully_connected(inputs=sequence, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                                           biases_initializer=tf.zeros_initializer(),
                                                           activation_fn=tf.nn.relu)
            projection = tf.nn.dropout(projection, keep_prob=self.dropout_keep_prob)
            projection = tf.contrib.layers.fully_connected(inputs=projection, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                                           biases_initializer=tf.zeros_initializer(),
                                                           activation_fn=tf.nn.relu)
            projection = tf.nn.dropout(projection, keep_prob=self.dropout_keep_prob)
        return projection

    def _transform_compare(self, sequence, reuse=False):
        with tf.variable_scope('transform_compare', reuse=reuse) as _:
            projection = tf.contrib.layers.fully_connected(inputs=sequence, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                                           biases_initializer=tf.zeros_initializer(),
                                                           activation_fn=tf.nn.relu)
            projection = tf.nn.dropout(projection, keep_prob=self.dropout_keep_prob)
            projection = tf.contrib.layers.fully_connected(inputs=projection, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                                           biases_initializer=tf.zeros_initializer(),
                                                           activation_fn=tf.nn.relu)
            projection = tf.nn.dropout(projection, keep_prob=self.dropout_keep_prob)
        return projection
