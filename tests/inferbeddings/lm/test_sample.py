# -*- coding: utf-8 -*-

import pytest

import json
import pickle

import tensorflow as tf

from inferbeddings.lm.model import LanguageModel

import logging

logger = logging.getLogger(__name__)


def test_lm_sample():
    checkpoint_path = 'models/snli/dam_1/dam_1'
    vocabulary_path = 'models/snli/dam_1/dam_1_index_to_token.p'
    lm_path = 'models/lm/'

    with open(vocabulary_path, 'rb') as f:
        index_to_token = pickle.load(f)

    index_to_token.update({
        0: '<PAD>',
        1: '<BOS>',
        2: '<UNK>'
    })

    token_to_index = {token: index for index, token in index_to_token.items()}

    vocab_size = len(token_to_index)
    embedding_size = 300

    print('vocab_size: {}'.format(vocab_size))

    with open('{}/config.json'.format(lm_path), 'r') as f:
        config = json.load(f)

    discriminator_scope_name = 'discriminator'
    with tf.variable_scope(discriminator_scope_name):
        embedding_layer = tf.get_variable('embeddings',
                                          shape=[vocab_size, embedding_size],
                                          initializer=tf.contrib.layers.xavier_initializer(),
                                          trainable=False)

    lm_scope_name = 'language_model'
    with tf.variable_scope(lm_scope_name) as _:
        imodel = LanguageModel(model=config['model'],
                               seq_length=config['seq_length'],
                               batch_size=config['batch_size'],
                               rnn_size=config['rnn_size'],
                               num_layers=config['num_layers'],
                               vocab_size=config['vocab_size'],
                               embedding_layer=embedding_layer,
                               infer=True)

    saver = tf.train.Saver(tf.global_variables())
    emb_saver = tf.train.Saver([embedding_layer], max_to_keep=1)

    with tf.Session() as session:
        emb_saver.restore(session, checkpoint_path)

        ckpt = tf.train.get_checkpoint_state(lm_path)
        saver.restore(session, ckpt.model_checkpoint_path)

        for _ in range(32):
            sample_value = imodel.sample(session=session,
                                         words=index_to_token,
                                         vocab=token_to_index,
                                         num=20,
                                         prime='. A',
                                         pick=1,
                                         sampling_type=1,
                                         width=4)
            print(sample_value)

if __name__ == '__main__':
    #pytest.main([__file__])
    test_lm_sample()
