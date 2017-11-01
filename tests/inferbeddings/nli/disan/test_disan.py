# -*- coding: utf-8 -*-

import tensorflow as tf

from inferbeddings.nli import tfutil
from inferbeddings.nli.disan.integration_func import traditional_attention


def test_disan():
    vocab_size, embedding_size = 10, 5

    sentence_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence')
    sentence_len_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence_length')

    embedding_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
    embedding_layer = tf.get_variable('embeddings', shape=[vocab_size, embedding_size],
                                      initializer=embedding_initializer)

    clipped_sentence = tfutil.clip_sentence(sentence_ph, sentence_len_ph)
    sentence_embedding = tf.nn.embedding_lookup(embedding_layer, clipped_sentence)

    sentence_representation = traditional_attention(
        sentence_embedding, tf.cast(True, tf.bool), 'traditional_attention',
        0.5, tf.cast(False, tf.bool), 5e-5)

    init_op = tf.global_variables_initializer()

    sentence_value = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    sentence_len_value = [4]

    with tf.Session() as session:
        session.run(init_op)

        feed_dict = {
            sentence_ph: sentence_value,
            sentence_len_ph: sentence_len_value
        }

        res = session.run(sentence_embedding, feed_dict=feed_dict)
        assert res.shape == (1, sentence_len_value[0], embedding_size)

        res = session.run(sentence_representation, feed_dict=feed_dict)
        assert res.shape == (1, embedding_size)


if __name__ == '__main__':
    test_disan()
