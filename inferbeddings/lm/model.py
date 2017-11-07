# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

from inferbeddings.lm.beam import BeamSearch

import logging

logger = logging.getLogger(__name__)


class LanguageModel:
    def __init__(self, model='rnn', seq_length=25, batch_size=50, rnn_size=256, num_layers=1,
                 vocab_size=None, infer=False, seed=0):

        assert vocab_size is not None

        if infer:
            batch_size = 1
            seq_length = 1

        cell_to_fn = {
            'rnn': rnn.BasicRNNCell,
            'gru': rnn.GRUCell,
            'lstm': rnn.BasicLSTMCell
        }

        if model not in cell_to_fn:
            raise ValueError("model type not supported: {}".format(model))

        cell_fn = cell_to_fn[model]
        cells = [cell_fn(rnn_size) for _ in range(num_layers)]

        self.cell = cell = rnn.MultiRNNCell(cells)

        self.input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
        self.targets = tf.placeholder(tf.int32, [batch_size, seq_length])
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [rnn_size, vocab_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
            softmax_b = tf.get_variable("softmax_b", [vocab_size], initializer=tf.zeros_initializer())

            embedding = tf.get_variable("embedding", [vocab_size, rnn_size])

            inputs = tf.split(tf.nn.embedding_lookup(embedding, self.input_data), seq_length, 1)
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, rnn_size])

        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probabilities = tf.nn.softmax(self.logits)

        loss = legacy_seq2seq.sequence_loss_by_example([self.logits],
                                                       [tf.reshape(self.targets, [-1])],
                                                       [tf.ones([batch_size * seq_length])],
                                                       vocab_size)

        self.cost = tf.reduce_sum(loss) / batch_size / seq_length
        self.final_state = last_state

        self.random_state = np.random.RandomState(seed)

    def sample(self, session, words, vocab, num=200, prime='first all', sampling_type=1, pick=0, width=4):
        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return int(np.searchsorted(t, np.random.rand(1) * s))

        def beam_search_predict(sample, state):
            """Returns the updated probability distribution (`probs`) and
            `state` for a given `sample`. `sample` should be a sequence of
            vocabulary labels, with the last word to be tested against the RNN.
            """
            x = np.zeros((1, 1))
            x[0, 0] = sample[-1]

            feed_dict = {
                self.input_data: x,
                self.initial_state: state
            }
            probabilities, final_state = session.run([self.probabilities, self.final_state], feed_dict=feed_dict)

            return probabilities, final_state

        def beam_search_pick(prime, width):
            """Returns the beam search pick."""
            if not len(prime) or prime == ' ':
                prime = self.random_state.choice(list(vocab.keys()))

            prime_labels = [vocab.get(word, 0) for word in prime.split()]
            bs = BeamSearch(beam_search_predict, session.run(self.cell.zero_state(1, tf.float32)), prime_labels)
            samples, scores = bs.search(None, None, k=width, maxsample=num)
            return samples[np.argmin(scores)]

        res = ''
        if pick == 1:
            state = session.run(self.cell.zero_state(1, tf.float32))
            if not len(prime) or prime == ' ':
                prime = self.random_state.choice(list(vocab.keys()))

            logger.info('Prime: {}'.format(prime))

            for word in prime.split()[:-1]:
                logger.info('Word: {}'.format(word))
                x = np.zeros((1, 1))
                x[0, 0] = vocab.get(word, 0)
                feed = {
                    self.input_data: x,
                    self.initial_state: state
                }
                state = session.run([self.final_state], feed)

            res = prime
            word = prime.split()[-1]

            for n in range(num):
                x = np.zeros((1, 1))
                x[0, 0] = vocab.get(word, 0)
                feed = {
                    self.input_data: x,
                    self.initial_state: state
                }
                probabilities, state = session.run([self.probabilities, self.final_state], feed)
                p = probabilities[0]

                if sampling_type == 0:
                    sample = np.argmax(p)
                elif sampling_type == 2:
                    sample = weighted_pick(p) if word == '\n' else np.argmax(p)
                else:
                    sample = weighted_pick(p)

                predictions = words[sample]
                res += ' ' + predictions
                word = predictions
        elif pick == 2:
            predictions = beam_search_pick(prime, width)
            for i, label in enumerate(predictions):
                res += ' ' + words[label] if i > 0 else words[label]
        return res
