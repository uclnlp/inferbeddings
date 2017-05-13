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

        sentence = [[1, 2, 3], [1, 2, 3]]
        sentence_length = [3, 3]

        feed_dict = {
            model.sentence1: sentence, model.sentence2: sentence,
            model.sentence1_size: sentence_length, model.sentence2_size: sentence_length}

        embedded1_value = session.run(model.embedded1, feed_dict=feed_dict)
        embedded2_value = session.run(model.embedded2, feed_dict=feed_dict)

        assert embedded1_value.shape == embedded2_value.shape == (2, sentence_length[0], representation_size)

        np.testing.assert_allclose(embedded1_value, embedded2_value)
        np.testing.assert_allclose(embedded1_value[0], embedded1_value[1])

        raw_attentions_value = session.run(model.raw_attentions, feed_dict=feed_dict)
        assert raw_attentions_value.shape == (2, sentence_length[0], sentence_length[0])
        np.testing.assert_allclose(raw_attentions_value[0], raw_attentions_value[1])

        attention_sentence1_value, attention_sentence2_value =\
            session.run([model.attention_sentence1, model.attention_sentence1], feed_dict=feed_dict)
        np.testing.assert_allclose(attention_sentence1_value[0], attention_sentence1_value[1])
        np.testing.assert_allclose(attention_sentence2_value[0], attention_sentence2_value[1])

        alpha_value, beta_value = session.run([model.alpha, model.beta], feed_dict=feed_dict)
        assert alpha_value.shape == beta_value.shape == (2, sentence_length[0], representation_size)
        np.testing.assert_allclose(alpha_value[0], alpha_value[1])

        v1_value, v2_value = session.run([model.v1, model.v2], feed_dict=feed_dict)
        assert v1_value.shape == v2_value.shape == (2, sentence_length[0], representation_size)
        np.testing.assert_allclose(v1_value[0], v1_value[1])

        logits_value = session.run(model.logits, feed_dict=feed_dict)
        assert logits_value.shape == (2, 3)
        np.testing.assert_allclose(logits_value[0], logits_value[1])


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    pytest.main([__file__])
