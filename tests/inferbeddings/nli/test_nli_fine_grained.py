# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from inferbeddings.nli import FeedForwardDAMP

import logging

import pytest

logger = logging.getLogger(__name__)


@pytest.mark.light
def test_nli_dam_fine_grained():
    embedding_size = 50
    representation_size = 6
    vocab_size = 1024

    sentence1_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence1')
    sentence2_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence2')

    sentence1_length_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence1_length')
    sentence2_length_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence2_length')

    dropout_keep_prob_ph = tf.placeholder(tf.float32, name='dropout_keep_prob')

    embedding_layer = tf.get_variable('embeddings', shape=[vocab_size, embedding_size],
                                      initializer=tf.contrib.layers.xavier_initializer())

    sentence1_embedding = tf.nn.embedding_lookup(embedding_layer, sentence1_ph)
    sentence2_embedding = tf.nn.embedding_lookup(embedding_layer, sentence2_ph)

    model_kwargs = dict(
        sequence1=sentence1_embedding, sequence1_length=sentence1_length_ph,
        sequence2=sentence2_embedding, sequence2_length=sentence2_length_ph,
        representation_size=representation_size, dropout_keep_prob=dropout_keep_prob_ph,
        use_masking=False)
    model_class = FeedForwardDAMP

    model = model_class(**model_kwargs)

    logits = model()

    # restore_path = 'models/nli/damp_v1.ckpt'
    init_op = tf.global_variables_initializer()

    sentence1 = [1, 2, 3, 4, 5]
    sentence2 = [6, 7, 8, 9]

    sentence1_length = 5
    sentence2_length = 4

    with tf.Session() as session:
        # saver = tf.train.Saver()
        # saver.restore(session, restore_path)
        session.run(init_op)

        feed_dict = {
            sentence1_ph: [sentence1], sentence1_length_ph: [sentence1_length],
            sentence2_ph: [sentence2], sentence2_length_ph: [sentence2_length],
            dropout_keep_prob_ph: 1.0
        }

        transformed_sequence1_value = session.run(model.transformed_sequence1, feed_dict=feed_dict)
        transformed_sequence2_value = session.run(model.transformed_sequence2, feed_dict=feed_dict)

        assert transformed_sequence1_value.shape == (1, len(sentence1), representation_size)
        assert transformed_sequence2_value.shape == (1, len(sentence2), representation_size)

        a_transformed_sequence1_value = session.run(model.attend_transformed_sequence1, feed_dict=feed_dict)
        a_transformed_sequence2_value = session.run(model.attend_transformed_sequence2, feed_dict=feed_dict)

        assert a_transformed_sequence1_value.shape == (1, len(sentence1), representation_size)
        assert a_transformed_sequence2_value.shape == (1, len(sentence2), representation_size)

        raw_attentions_value = session.run(model.raw_attentions, feed_dict=feed_dict)
        assert raw_attentions_value.shape == (1, len(sentence1), len(sentence2))

        _a = a_transformed_sequence1_value[0, :, :]
        _b = a_transformed_sequence2_value[0, :, :]
        _c = np.matmul(_a, np.transpose(_b, (1, 0)))

        print(_c.shape)

        np.testing.assert_allclose(_c, raw_attentions_value[0, :, :], rtol=1e-3)

        attention_sentence1_value = session.run(model.attention_sentence1, feed_dict=feed_dict)
        attention_sentence2_value = session.run(model.attention_sentence2, feed_dict=feed_dict)

        print(attention_sentence1_value)
        print(attention_sentence2_value)

        alpha_value, beta_value = session.run([model.alpha, model.beta], feed_dict=feed_dict)

        logits_value = session.run(logits, feed_dict=feed_dict)
        assert logits_value.shape == (1, 3)

    tf.reset_default_graph()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    pytest.main([__file__])
