# -*- coding: utf-8 -*-

import tensorflow as tf

import logging

logger = logging.getLogger(__name__)


class ConditionalBiLSTM:
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

        self.cell_fw = tf.contrib.rnn.LSTMCell(num_units, state_is_tuple=True,
                                               initializer=tf.contrib.layers.xavier_initializer())

        self.cell_fw = tf.contrib.rnn.DropoutWrapper(self.cell_fw, input_keep_prob=dropout_keep_prob,
                                                     output_keep_prob=dropout_keep_prob)

        self.cell_bw = tf.contrib.rnn.LSTMCell(num_units, state_is_tuple=True,
                                               initializer=tf.contrib.layers.xavier_initializer())

        self.cell_bw = tf.contrib.rnn.DropoutWrapper(self.cell_bw, input_keep_prob=dropout_keep_prob,
                                                     output_keep_prob=dropout_keep_prob)

        # [batch, O(num_unit)]
        self.encoded_sequence1, output_state =\
            self._bidirectional_encoder(inputs=self.embedded1, sequence_length=self.sentence1_size)

        # [batch, O(num_unit)]
        self.encoded_sequence2, _ =\
            self._bidirectional_encoder(inputs=self.embedded2, sequence_length=self.sentence2_size,
                                        initial_state=output_state, reuse=True)

        # [batch, O(num_units)]
        self.pre_logits = tf.concat(values=[self.encoded_sequence1, self.encoded_sequence2], axis=1)

        # [batch, 3]
        self.logits = tf.contrib.layers.fully_connected(inputs=self.pre_logits, num_outputs=num_classes,
                                                        weights_initializer=tf.random_normal_initializer(0.0, 0.1),
                                                        biases_initializer=tf.zeros_initializer(), activation_fn=None)
        self.logits = tf.contrib.layers.dropout(self.logits, keep_prob=dropout_keep_prob)

        self.predictions = tf.argmax(self.logits, axis=1, name='predictions')
        labels = tf.one_hot(self.label, num_classes)
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

    def _bidirectional_encoder(self, inputs, sequence_length=None,
                               initial_state=None, reuse=False):

        initial_state_fw, initial_state_bw = None, None
        if initial_state is not None:
            initial_state_fw, initial_state_bw = initial_state

        with tf.variable_scope('encoder', reuse=reuse):
            outputs, (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self.cell_fw, cell_bw=self.cell_bw,
                initial_state_fw=initial_state_fw,
                initial_state_bw=initial_state_bw,
                inputs=inputs,
                sequence_length=sequence_length,
                dtype=tf.float32)

        encoded = tf.concat(values=[output_state_fw.h, output_state_bw.h], axis=1)
        return encoded, (output_state_fw, output_state_bw)
