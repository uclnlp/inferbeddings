# -*- coding: utf-8 -*-

import tensorflow as tf

from inferbeddings.rte.dam import FeedForwardDAM
from inferbeddings.rte.util import count_parameters
import logging

logger = logging.getLogger(__name__)


import pytest


def test_ff_dam():
    vocab_size = 1000
    embedding_size = 32
    representation_size = 16,
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
        logger.debug('Total parameters: {}'.format(count_parameters()))

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    pytest.main([__file__])
