# -*- coding: utf-8 -*-

import tensorflow as tf

from inferbeddings.rte2 import BaseRTEModel

import logging

logger = logging.getLogger(__name__)


class ConditionalBiLSTM(BaseRTEModel):
    def __init__(self, hidden_size=100, dropout_keep_prob=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden_size = hidden_size
        self.dropout_keep_prob = dropout_keep_prob

        with tf.variable_scope('lstm', reuse=self.reuse) as _:
            self.cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size, state_is_tuple=True,
                                                   initializer=tf.contrib.layers.xavier_initializer())
            if self.dropout_keep_prob:
                self.cell_fw = tf.contrib.rnn.DropoutWrapper(self.cell_fw, input_keep_prob=self.dropout_keep_prob,
                                                             output_keep_prob=self.dropout_keep_prob)

            self.cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size, state_is_tuple=True,
                                                   initializer=tf.contrib.layers.xavier_initializer())
            if self.dropout_keep_prob:
                self.cell_bw = tf.contrib.rnn.DropoutWrapper(self.cell_bw, input_keep_prob=self.dropout_keep_prob,
                                                             output_keep_prob=self.dropout_keep_prob)

            self.encoded_sequence1, output_state = \
                self._encoder(sequence=self.sequence1, sequence_length=self.sequence1_length, reuse=self.reuse)

            self.encoded_sequence2, _ = \
                self._encoder(sequence=self.sequence2, sequence_length=self.sequence2_length,
                              initial_state=output_state, reuse=True)

        self.encoded_sequences = tf.concat(values=[self.encoded_sequence1, self.encoded_sequence2], axis=1)

        with tf.variable_scope('output', reuse=self.reuse) as _:
            self.logits = tf.contrib.layers.fully_connected(inputs=self.encoded_sequences, num_outputs=self.nb_classes,
                                                            weights_initializer=tf.random_normal_initializer(0.0, 0.1),
                                                            biases_initializer=tf.zeros_initializer(),
                                                            activation_fn=None)

    def __call__(self):
            return self.logits

    def _encoder(self, sequence, sequence_length=None, initial_state=None, reuse=False):
        initial_state_fw = initial_state_bw = initial_state

        with tf.variable_scope('encoder', reuse=reuse or self.reuse) as _:
            outputs, (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self.cell_fw, cell_bw=self.cell_bw,
                initial_state_fw=initial_state_fw,
                initial_state_bw=initial_state_bw,
                inputs=sequence,
                sequence_length=sequence_length,
                dtype=tf.float32)

        encoded = tf.concat(values=[output_state_fw.h, output_state_bw.h], axis=1)
        return encoded, (output_state_fw, output_state_bw)
