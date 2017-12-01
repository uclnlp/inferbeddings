#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

from inferbeddings.nli import tfutil

import pickle
import json

import logging

logger = logging.getLogger(__name__)


def main(argv):
    vocabulary_path = 'models/snli/dam_1/dam_1_index_to_token.p'
    checkpoint_path = 'models/snli/dam_1/dam_1'
    lm_path = 'models/lm/'

    with open(vocabulary_path, 'rb') as f:
        index_to_token = pickle.load(f)

    index_to_token.update({
        0: '<PAD>',
        1: '<BOS>',
        2: '<UNK>'
    })

    token_to_index = {token: index for index, token in index_to_token.items()}

    with open('{}/config.json'.format(lm_path), 'r') as f:
        config = json.load(f)

    seq_length = config['seq_length']
    batch_size = 1  # config['batch_size']
    rnn_size = config['rnn_size']
    num_layers = config['num_layers']

    vocab_size = len(token_to_index)
    assert vocab_size == config['vocab_size']

    discriminator_scope_name = 'discriminator'
    with tf.variable_scope(discriminator_scope_name):
        embedding_layer = tf.get_variable('embeddings',
                                          shape=[vocab_size, config['embedding_size']],
                                          initializer=tf.contrib.layers.xavier_initializer(),
                                          trainable=False)

    lm_scope_name = 'language_model'
    with tf.variable_scope(lm_scope_name):
        cell_fn = rnn.BasicLSTMCell
        cells = [cell_fn(rnn_size) for _ in range(num_layers)]

        cell = rnn.MultiRNNCell(cells)

        input_data = tf.placeholder(tf.int32, [None, seq_length])
        targets = tf.placeholder(tf.int32, [None, seq_length])
        initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            W = tf.get_variable(name='W',
                                shape=[rnn_size, vocab_size],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='b',
                                shape=[vocab_size],
                                initializer=tf.zeros_initializer())

            emb_lookup = tf.nn.embedding_lookup(embedding_layer, input_data)
            emb_projection = tf.contrib.layers.fully_connected(inputs=emb_lookup,
                                                               num_outputs=rnn_size,
                                                               weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                               biases_initializer=tf.zeros_initializer())

            inputs = tf.split(emb_projection, seq_length, 1)
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        outputs, last_state = legacy_seq2seq.rnn_decoder(decoder_inputs=inputs,
                                                         initial_state=initial_state,
                                                         cell=cell,
                                                         loop_function=None,
                                                         scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, rnn_size])

        logits = tf.matmul(output, W) + b
        probabilities = tf.nn.softmax(logits)

        loss = legacy_seq2seq.sequence_loss_by_example(logits=[logits],
                                                       targets=[tf.reshape(targets, [-1])],
                                                       weights=[tf.ones([batch_size * seq_length])])

        cost = tf.reduce_sum(loss) / batch_size / seq_length
        final_state = last_state

    saver = tf.train.Saver(tf.global_variables())
    emb_saver = tf.train.Saver([embedding_layer], max_to_keep=1)

    logger.info('Creating the session ..')

    with tf.Session() as session:
        emb_saver.restore(session, checkpoint_path)

        ckpt = tf.train.get_checkpoint_state(lm_path)
        assert ckpt is not None
        saver.restore(session, ckpt.model_checkpoint_path)

        logger.info('Total Parameters: {}'.format(tfutil.count_trainable_parameters()))

        state = session.run(cell.zero_state(batch_size, tf.float32))

        x = np.zeros(shape=(1, seq_length))
        y = np.zeros(shape=(1, seq_length))

        x[0, :] = [token_to_index[t] for t in ['A', 'happy', 'girl', 'runs', 'with', 'a', 'red', 'ball']]
        y[0, :] = [token_to_index[t] for t in ['happy', 'girl', 'runs', 'with', 'a', 'red', 'ball', '.']]

        fd = {
            input_data: x,
            targets: y,
            initial_state: state
        }

        cost_value, state = session.run([cost, final_state], feed_dict=fd)

        print(cost_value)






if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
