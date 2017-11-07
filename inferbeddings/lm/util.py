# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from inferbeddings.io import load_glove, load_glove_words


def create_embedding_matrix(vocab_size, rnn_size, is_trainable=True):
    embedding = tf.get_variable("embedding", [vocab_size, rnn_size], trainable=is_trainable,
                                initializer=tf.contrib.layers.xavier_initializer())
    return embedding
