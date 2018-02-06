# -*- coding: utf-8 -*-

import pytest

import os
import pickle
import json

import numpy as np
import tensorflow as tf

from inferbeddings.nli import tfutil
from inferbeddings.nli.generate.scorer import LMScorer


def test_scorer():
    vocabulary_path = 'models/snli/dam_1/dam_1_index_to_token.p'
    checkpoint_path = 'models/snli/dam_1/dam_1'
    lm_path = 'models/lm/'

    batch_size = 128

    with open(vocabulary_path, 'rb') as f:
        index_to_token = pickle.load(f)

    index_to_token.update({
        0: '<PAD>',
        1: '<BOS>',
        2: '<UNK>'
    })

    token_to_index = {token: index for index, token in index_to_token.items()}

    with open(os.path.join(lm_path, 'config.json'), 'r') as f:
        config = json.load(f)

    vocab_size = len(token_to_index)

    discriminator_scope_name = 'discriminator'
    with tf.variable_scope(discriminator_scope_name):
        embedding_layer = tf.get_variable('embeddings',
                                          shape=[vocab_size, config['embedding_size']],
                                          initializer=tf.contrib.layers.xavier_initializer(),
                                          trainable=False)

    generator = LMScorer(embedding_layer=embedding_layer, token_to_index=token_to_index, batch_size=batch_size)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    discriminator_vars = tfutil.get_variables_in_scope(discriminator_scope_name)
    lm_vars = generator.get_lm_vars()

    saver = tf.train.Saver(discriminator_vars, max_to_keep=1)
    lm_saver = tf.train.Saver(lm_vars, max_to_keep=1)

    with tf.Session(config=session_config) as session:
        saver.restore(session, checkpoint_path)

        lm_ckpt = tf.train.get_checkpoint_state(lm_path)
        lm_saver.restore(session, lm_ckpt.model_checkpoint_path)

        bos_idx, eos_idx, unk_idx = 1, 2, 3

        sentence = 'The girl runs on the grass .'
        sentence_idx = [bos_idx] + [token_to_index[token] for token in sentence.split()]

        np_sentence_idxs = np.array([sentence_idx] * batch_size)
        np_sentence_lens = np.array([len(sentence_idx)] * batch_size)

        v = generator.log_perplexity(session, np_sentence_idxs, np_sentence_lens)
        for e in v.tolist():
            assert abs(e - 15.976228) < 1e-4

if __name__ == '__main__':
    # pytest.main([__file__])
    test_scorer()
