# -*- coding: utf-8 -*-

import pytest

import pickle

import tensorflow as tf

from inferbeddings.lm.loader2 import SNLILoader
from inferbeddings.lm.model import LanguageModel

import logging

logger = logging.getLogger(__name__)


def test_lm_snli_overfit():
    checkpoint_path = 'models/snli/dam_1/dam_1'
    vocabulary_path = 'models/snli/dam_1/dam_1_index_to_token.p'

    with open(vocabulary_path, 'rb') as f:
        index_to_token = pickle.load(f)
    token_to_index = {token: index for index, token in index_to_token.items()}

    vocab_size = len(token_to_index)
    embedding_size = 300
    rnn_size = 64
    num_epochs = 10

    learning_rate = 0.1

    config = {
        'model': 'lstm',
        'seq_length': 4,
        'batch_size': 8,
        'vocab_size': vocab_size,
        'embedding_size': embedding_size,
        'rnn_size': rnn_size,
        'num_layers': 1
    }

    loader = SNLILoader(path='data/snli/tiny/one.jsonl.gz',
                        token_to_index=token_to_index,
                        batch_size=config['batch_size'],
                        seq_length=config['seq_length'],
                        shuffle=True)

    loader.create_batches()

    discriminator_scope_name = 'discriminator'
    with tf.variable_scope(discriminator_scope_name):
        embedding_layer = tf.get_variable('embeddings',
                                          shape=[vocab_size + 3, embedding_size],
                                          initializer=tf.contrib.layers.xavier_initializer(),
                                          trainable=False)

    lm_scope_name = 'language_model'
    with tf.variable_scope(lm_scope_name) as scope:
        model = LanguageModel(model=config['model'],
                              seq_length=config['seq_length'],
                              batch_size=config['batch_size'],
                              rnn_size=config['rnn_size'],
                              num_layers=config['num_layers'],
                              vocab_size=config['vocab_size'],
                              embedding_layer=embedding_layer,
                              infer=False)

        scope.reuse_variables()
        imodel = LanguageModel(model=config['model'],
                               seq_length=config['seq_length'],
                               batch_size=config['batch_size'],
                               rnn_size=config['rnn_size'],
                               num_layers=config['num_layers'],
                               vocab_size=config['vocab_size'],
                               embedding_layer=embedding_layer,
                               infer=True)

    optimizer = tf.train.AdagradOptimizer(learning_rate)
    train_op = optimizer.minimize(model.cost)

    init_op = tf.global_variables_initializer()

    emb_saver = tf.train.Saver([embedding_layer], max_to_keep=1)

    with tf.Session() as session:
        session.run(init_op)

        emb_saver.restore(session, checkpoint_path)

        for epoch_id in range(0, num_epochs):
            logger.debug('Epoch: {}'.format(epoch_id))

            loader.reset_batch_pointer()
            state = session.run(model.initial_state)

            for batch_id in range(loader.pointer, loader.num_batches):
                x, y = loader.next_batch()

                feed_dict = {
                    model.input_data: x,
                    model.targets: y,
                    model.initial_state: state
                }

                loss_value = session.run(model.cost, feed_dict=feed_dict)
                state = session.run(model.final_state, feed_dict=feed_dict)
                _ = session.run(train_op, feed_dict=feed_dict)

                print(loss_value)

                sample_value = imodel.sample(session, index_to_token, token_to_index,
                                             10, 'A', 0, 1, 4)
                print(sample_value)

if __name__ == '__main__':
    pytest.main([__file__])
