# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from inferbeddings.rte.dam import FeedForwardDAM
from inferbeddings.rte.util import count_parameters
import logging

logger = logging.getLogger(__name__)


import pytest


def test_ff_dam():
    vocab_size = 1000
    embedding_size = 32
    representation_size = 16
    dropout_keep_prob = 1.0
    is_fixed_embeddings = False
    learning_rate = 0.1

    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)

    kwargs = dict(optimizer=optimizer, vocab_size=vocab_size, embedding_size=embedding_size,
        representation_size=representation_size, dropout_keep_prob=dropout_keep_prob,
        l2_lambda=None, trainable_embeddings=not is_fixed_embeddings)

    model = FeedForwardDAM(**kwargs)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    init_op = tf.global_variables_initializer()

    with tf.Session(config=session_config) as session:
        session.run(init_op)
        nb_parameters = count_parameters()

        sentence = [[1, 2, 3]]
        sentence_length = [3]

        feed_dict = {
            model.sentence1: sentence, model.sentence2: sentence,
            model.sentence1_size: sentence_length, model.sentence2_size: sentence_length}

        embedded1_value = session.run(model.embedded1, feed_dict=feed_dict)
        embedded2_value = session.run(model.embedded2, feed_dict=feed_dict)

        np.testing.assert_allclose(embedded1_value, embedded2_value)

        alpha_value, beta_value = embedded2_value = session.run([model.alpha, model.beta], feed_dict=feed_dict)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    pytest.main([__file__])
