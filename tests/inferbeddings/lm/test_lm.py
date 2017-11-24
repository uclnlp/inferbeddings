# -*- coding: utf-8 -*-

import argparse

import pickle

import numpy as np
import tensorflow as tf

import pytest

from inferbeddings.lm.loader import SNLILoader
from inferbeddings.lm.model import LanguageModel
from inferbeddings.nli import tfutil

import logging

logger = logging.getLogger(__name__)


def stats(values):
    return '{0:.4f} ± {1:.4f}'.format(round(np.mean(values), 4), round(np.std(values), 4))


def test_lm_overfit():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='data/snli/snli_1.0_test.jsonl.gz')

    parser.add_argument('--vocabulary', type=str, default='models/snli/dam_1/dam_1_index_to_token.p')
    parser.add_argument('--checkpoint', type=str, default='models/snli/dam_1/dam_1')

    parser.add_argument('--save', type=str, default='./models/lm/', help='directory to store checkpointed models')

    parser.add_argument('--embedding-size', type=int, default=300, help='embedding size')
    parser.add_argument('--rnn-size', type=int, default=256, help='size of RNN hidden state')
    parser.add_argument('--num-layers', type=int, default=1, help='number of layers in the RNN')

    parser.add_argument('--model', type=str, default='lstm', help='rnn, gru, or lstm')

    parser.add_argument('--batch-size', type=int, default=128, help='minibatch size')
    parser.add_argument('--seq-length', type=int, default=8, help='RNN sequence length')
    parser.add_argument('--num-epochs', type=int, default=100, help='number of epochs')

    parser.add_argument('--report-every', '-r', type=int, default=10, help='report loss frequency')
    parser.add_argument('--save-every', '-s', type=int, default=100, help='save frequency')

    parser.add_argument('--grad-clip', type=float, default=5., help='clip gradients at this value')
    parser.add_argument('--learning-rate', '--lr', type=float, default=0.001, help='learning rate')

    args = parser.parse_args('')

    vocabulary_path = args.vocabulary
    checkpoint_path = args.checkpoint

    with open(vocabulary_path, 'rb') as f:
        index_to_token = pickle.load(f)

    token_to_index = {token: index for index, token in index_to_token.items()}

    logger.info('Loading the dataset ..')

    loader = SNLILoader(path=args.data,
                        token_to_index=token_to_index,
                        batch_size=args.batch_size,
                        seq_length=args.seq_length)
    vocab_size = len(token_to_index)

    config = {
        'model': args.model,
        'seq_length': args.seq_length,
        'batch_size': args.batch_size,
        'vocab_size': vocab_size,
        'embedding_size': args.embedding_size,
        'rnn_size': args.rnn_size,
        'num_layers': args.num_layers
    }

    logger.info('Generating the computational graph ..')

    discriminator_scope_name = 'discriminator'
    with tf.variable_scope(discriminator_scope_name):
        embedding_layer = tf.get_variable('embeddings',
                                          shape=[vocab_size + 3, args.embedding_size],
                                          initializer=tf.contrib.layers.xavier_initializer(),
                                          trainable=False)

    lm_scope_name = 'language_model'
    with tf.variable_scope(lm_scope_name):
        model = LanguageModel(model=config['model'],
                              seq_length=config['seq_length'],
                              batch_size=config['batch_size'],
                              rnn_size=config['rnn_size'],
                              num_layers=config['num_layers'],
                              vocab_size=config['vocab_size'],
                              embedding_layer=embedding_layer,
                              infer=False)

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(model.cost, tvars), args.grad_clip)

    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    init_op = tf.global_variables_initializer()

    emb_saver = tf.train.Saver([embedding_layer], max_to_keep=1)

    logger.info('Creating the session ..')

    with tf.Session(config=session_config) as session:
        logger.info('Total Parameters: {}'.format(tfutil.count_trainable_parameters()))
        session.run(init_op)

        emb_saver.restore(session, checkpoint_path)

        loss_values = []

        for epoch_id in range(0, args.num_epochs):
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

                loss_value, state, _ = session.run([model.cost, model.final_state, train_op], feed_dict=feed_dict)
                loss_values += [loss_value]

                if (epoch_id * loader.num_batches + batch_id) % args.report_every == 0:
                    a = epoch_id * loader.num_batches + batch_id
                    b = args.num_epochs * loader.num_batches
                    logger.info("{}/{} (epoch {}), loss = {}".format(a, b, epoch_id, stats(loss_values)))
                    loss_values = []

if __name__ == '__main__':
    pytest.main([__file__])
