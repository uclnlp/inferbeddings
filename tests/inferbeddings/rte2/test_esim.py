# -*- coding: utf-8 -*-

import sys

import numpy as np
import tensorflow as tf

from inferbeddings.rte2.esim import ESIMv1
import logging

import pytest

logger = logging.getLogger(__name__)


def count_trainable_parameters():
    """
    Count the number of trainable tensorflow parameters loaded in
    the current graph.
    """
    total_params = 0
    for variable in tf.trainable_variables():
        variable_params = np.prod([1] + [dim.value for dim in variable.get_shape()])
        print('{}: {} params'.format(variable.name, variable_params))
        total_params += variable_params
    return total_params


def test_esim():
    vocab_size = 1000
    embedding_size = 300
    representation_size = 300

    sentence1_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence1')
    sentence2_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence2')

    sentence1_length_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence1_length')
    sentence2_length_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence2_length')

    embedding_layer = tf.get_variable('embeddings', shape=[vocab_size, embedding_size],
                                      initializer=tf.contrib.layers.xavier_initializer())

    sentence1_embedding = tf.nn.embedding_lookup(embedding_layer, sentence1_ph)
    sentence2_embedding = tf.nn.embedding_lookup(embedding_layer, sentence2_ph)

    dropout_keep_prob_ph = 1.0

    model_kwargs = dict(
        sequence1=sentence1_embedding, sequence1_length=sentence1_length_ph,
        sequence2=sentence2_embedding, sequence2_length=sentence2_length_ph,
        representation_size=representation_size, dropout_keep_prob=dropout_keep_prob_ph)

    model = ESIMv1(**model_kwargs)

    init_op = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init_op)

        nb_parameters = count_trainable_parameters()

        seq_length = 110
        s = list(range(seq_length))
        sentence = [s, s]
        sentence_length = [50, 50]

        feed_dict = {
            sentence1_ph: sentence,
            sentence2_ph: sentence,
            sentence1_length_ph: sentence_length,
            sentence2_length_ph: sentence_length
        }

        embedded1_value = session.run(sentence1_embedding, feed_dict=feed_dict)
        embedded2_value = session.run(sentence2_embedding, feed_dict=feed_dict)

        assert embedded1_value.shape == embedded2_value.shape == (2, seq_length, embedding_size)

        np.testing.assert_allclose(embedded1_value, embedded2_value)
        np.testing.assert_allclose(embedded1_value[0], embedded1_value[1])

        transformed_sequence1 = session.run(model.transformed_sequence1, feed_dict=feed_dict)
        transformed_sequence2 = session.run(model.transformed_sequence2, feed_dict=feed_dict)

        assert transformed_sequence1.shape == transformed_sequence2.shape == (2, seq_length, representation_size * 2)

        raw_attentions_value = session.run(model.raw_attentions, feed_dict=feed_dict)
        assert raw_attentions_value.shape == (2, seq_length, seq_length)
        np.testing.assert_allclose(raw_attentions_value[0], raw_attentions_value[1])

        attention_sentence1_value, attention_sentence2_value =\
            session.run([model.attention_sentence1, model.attention_sentence1], feed_dict=feed_dict)
        np.testing.assert_allclose(attention_sentence1_value[0], attention_sentence1_value[1])
        np.testing.assert_allclose(attention_sentence2_value[0], attention_sentence2_value[1])

        alpha_value, beta_value = session.run([model.alpha, model.beta], feed_dict=feed_dict)
        assert alpha_value.shape == beta_value.shape == (2, seq_length, representation_size * 2)
        np.testing.assert_allclose(alpha_value[0], alpha_value[1])

        v1_value, v2_value = session.run([model.v1, model.v2], feed_dict=feed_dict)
        assert v1_value.shape == v2_value.shape == (2, seq_length, representation_size * 2)
        np.testing.assert_allclose(v1_value[0], v1_value[1])

        logits_value = session.run(model.logits, feed_dict=feed_dict)
        assert logits_value.shape == (2, 3)
        np.testing.assert_allclose(logits_value[0], logits_value[1])

    tf.reset_default_graph()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    pytest.main([__file__])
