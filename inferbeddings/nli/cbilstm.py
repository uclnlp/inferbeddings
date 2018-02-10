# -*- coding: utf-8 -*-

import tensorflow as tf

from inferbeddings.nli import BaseRTEModel

import logging

logger = logging.getLogger(__name__)


def fused_rnn_backward(fused_rnn, inputs, sequence_length, initial_state=None, dtype=None, scope=None, time_major=True):
    if not time_major:
        inputs = tf.transpose(inputs, [1, 0, 2])
    # assumes that time dim is 0 and batch is 1
    rev_inputs = tf.reverse_sequence(inputs, sequence_length, 0, 1)
    rev_outputs, last_state = fused_rnn(rev_inputs, sequence_length=sequence_length, initial_state=initial_state,
                                        dtype=dtype, scope=scope)
    outputs = tf.reverse_sequence(rev_outputs, sequence_length, 0, 1)
    if not time_major:
        outputs = tf.transpose(outputs, [1, 0, 2])
    return outputs, last_state


def fused_birnn(fused_rnn, inputs, sequence_length, initial_state=(None, None), dtype=None, scope=None,
                time_major=False, backward_device=None):
    with tf.variable_scope(scope or "BiRNN"):
        sequence_length = tf.cast(sequence_length, tf.int32)
        if not time_major:
            inputs = tf.transpose(inputs, [1, 0, 2])
        outputs_fw, state_fw = fused_rnn(inputs, sequence_length=sequence_length, initial_state=initial_state[0],
                                         dtype=dtype, scope="FW")

        if backward_device is not None:
            with tf.device(backward_device):
                outputs_bw, state_bw = fused_rnn_backward(fused_rnn, inputs, sequence_length, initial_state[1], dtype,
                                                          scope="BW")
        else:
            outputs_bw, state_bw = fused_rnn_backward(fused_rnn, inputs, sequence_length, initial_state[1], dtype,
                                                      scope="BW")

        if not time_major:
            outputs_fw = tf.transpose(outputs_fw, [1, 0, 2])
            outputs_bw = tf.transpose(outputs_bw, [1, 0, 2])
    return (outputs_fw, outputs_bw), (state_fw, state_bw)


class ConditionalBiLSTM(BaseRTEModel):
    def __init__(self, representation_size=300, dropout_keep_prob=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.representation_size = representation_size
        self.dropout_keep_prob = dropout_keep_prob

        with tf.variable_scope('lstm', reuse=self.reuse) as _:
            fused_rnn = tf.contrib.rnn.LSTMBlockFusedCell(self.representation_size)
            # [batch, 2*output_dim] -> [batch, num_classes]
            _, q_states = fused_birnn(fused_rnn, self.sequence1, sequence_length=self.sequence1_length,
                                      dtype=tf.float32, time_major=False, scope="sequence1_rnn")
            outputs, _ = fused_birnn(fused_rnn, self.sequence2, sequence_length=self.sequence2_length,
                                     dtype=tf.float32, initial_state=q_states, time_major=False, scope="sequence2_rnn")

            outputs = tf.concat([outputs[0], outputs[1]], axis=2)
            hidden = tf.layers.dense(outputs, self.representation_size, tf.nn.relu, name="hidden") * tf.expand_dims(
                tf.sequence_mask(self.sequence2_length, maxlen=tf.shape(outputs)[1], dtype=tf.float32), 2)
            hidden = tf.reduce_max(hidden, axis=1)
            # [batch, dim] -> [batch, num_classes]
            outputs = tf.layers.dense(hidden, self.nb_classes, name="classification")
            self.logits = outputs

    def __call__(self):
            return self.logits
