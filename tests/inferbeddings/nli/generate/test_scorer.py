# -*- coding: utf-8 -*-

import pytest

import os
import pickle
import json

import numpy as np
import tensorflow as tf

from inferbeddings.nli import FeedForwardDAM

from inferbeddings.nli import tfutil
from inferbeddings.nli.generate.scorer import LMScorer
from inferbeddings.nli.generate.scorer import IScorer


def test_i_scorer():
    tf.reset_default_graph()

    vocabulary_path = 'models/snli/dam_1/dam_1_index_to_token.p'
    checkpoint_path = 'models/snli/dam_1/dam_1'
    lm_path = 'models/lm/'

    batch_size = 128
    representation_size = 200

    with open(vocabulary_path, 'rb') as f:
        index_to_token = pickle.load(f)

    index_to_token.update({
        0: '<PAD>',
        1: '<BOS>',
        2: '<UNK>'
    })

    with open(os.path.join(lm_path, 'config.json'), 'r') as f:
        config = json.load(f)

    token_to_index = {token: index for index, token in index_to_token.items()}
    vocab_size = len(token_to_index)

    sentence1_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence1')
    sentence2_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence2')

    sentence1_len_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence1_length')
    sentence2_len_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence2_length')

    clipped_sentence1 = tfutil.clip_sentence(sentence1_ph, sentence1_len_ph)
    clipped_sentence2 = tfutil.clip_sentence(sentence2_ph, sentence2_len_ph)

    discriminator_scope_name = 'discriminator'
    with tf.variable_scope(discriminator_scope_name):
        embedding_layer = tf.get_variable('embeddings',
                                          shape=[vocab_size, config['embedding_size']],
                                          initializer=tf.contrib.layers.xavier_initializer(),
                                          trainable=False)

        sentence1_embedding = tf.nn.embedding_lookup(embedding_layer, clipped_sentence1)
        sentence2_embedding = tf.nn.embedding_lookup(embedding_layer, clipped_sentence2)

        dropout_keep_prob_ph = tf.placeholder(tf.float32, name='dropout_keep_prob')

        model_kwargs = dict(
            sequence1=sentence1_embedding, sequence1_length=sentence1_len_ph,
            sequence2=sentence2_embedding, sequence2_length=sentence2_len_ph,
            representation_size=representation_size, dropout_keep_prob=dropout_keep_prob_ph)

        model_class = FeedForwardDAM
        model = model_class(**model_kwargs)

        scorer = IScorer(embedding_layer=embedding_layer,
                         token_to_index=token_to_index,
                         model_class=model_class,
                         model_kwargs=model_kwargs,
                         i_pooling_function=tf.reduce_sum)

        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True

        discriminator_vars = tfutil.get_variables_in_scope(discriminator_scope_name)

        saver = tf.train.Saver(discriminator_vars, max_to_keep=1)

        with tf.Session(config=session_config) as session:
            saver.restore(session, checkpoint_path)

            bos_idx, eos_idx, unk_idx = 1, 2, 3

            sentence1 = "A man in a tank top fixing himself a hotdog ."
            sentence1_idx = [bos_idx] + [token_to_index[token] for token in sentence1.split()]

            sentence2 = "two girls were ther"
            sentence2_idx = [bos_idx] + [token_to_index[token] for token in sentence2.split()]

            np_sentence1_idxs = np.array([sentence1_idx] * batch_size)
            np_sentence1_lens = np.array([len(sentence1_idx)] * batch_size)

            np_sentence2_idxs = np.array([sentence2_idx] * batch_size)
            np_sentence2_lens = np.array([len(sentence2_idx)] * batch_size)

            v1 = scorer.score(session=session,
                              sentences1=np_sentence1_idxs, sizes1=np_sentence1_lens,
                              sentences2=np_sentence2_idxs, sizes2=np_sentence2_lens)

            v2 = scorer.score(session=session,
                              sentences1=np_sentence2_idxs, sizes1=np_sentence2_lens,
                              sentences2=np_sentence1_idxs, sizes2=np_sentence1_lens)

            for e in v1.tolist():
                assert abs(e - 0.0) < 1e-4

            for e in v2.tolist():
                assert abs(e - 0.99952203) < 1e-4


def test_lm_scorer():
    tf.reset_default_graph()

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
    lm_vars = generator.get_vars()

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

        sentence = 'The girl runs on the airplane .'
        sentence_idx = [bos_idx] + [token_to_index[token] for token in sentence.split()]

        np_sentence_idxs = np.array([sentence_idx] * batch_size)
        np_sentence_lens = np.array([len(sentence_idx)] * batch_size)

        v1 = generator.log_perplexity(session, np_sentence_idxs, np_sentence_lens)
        assert np.all(v1 > v)

if __name__ == '__main__':
    pytest.main([__file__])
