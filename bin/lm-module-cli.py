#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import pickle
import sys

import tensorflow as tf

from inferbeddings.lm.legacy.loader import TextLoader
from inferbeddings.lm.model import LanguageModel
from inferbeddings.nli import tfutil

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/lm/neuromancer/',
                        help='data directory containing input.txt')

    parser.add_argument('--vocabulary', type=str, default='models/snli/dam_1/dam_1_index_to_token.p')
    parser.add_argument('--checkpoint', type=str, default='models/snli/dam_1/dam_1')

    parser.add_argument('--save', type=str, default='save', help='directory to store checkpointed models')

    parser.add_argument('--embedding-size', type=int, default=300, help='embedding size')
    parser.add_argument('--rnn-size', type=int, default=256, help='size of RNN hidden state')
    parser.add_argument('--num-layers', type=int, default=1, help='number of layers in the RNN')

    parser.add_argument('--model', type=str, default='rnn', help='rnn, gru, or lstm')

    parser.add_argument('--batch-size', type=int, default=50, help='minibatch size')
    parser.add_argument('--seq-length', type=int, default=25, help='RNN sequence length')
    parser.add_argument('--num-epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--save-every', type=int, default=1000, help='save frequency')
    parser.add_argument('--grad-clip', type=float, default=5., help='clip gradients at this value')
    parser.add_argument('--learning-rate', '--lr', type=float, default=0.002, help='learning rate')

    args = parser.parse_args(argv)
    train(args)


def train(args):
    vocabulary_path = args.vocabulary
    checkpoint_path = args.checkpoint

    with open(vocabulary_path, 'rb') as f:
        index_to_token = pickle.load(f)

    token_to_index = {token: index for index, token in index_to_token.items()}

    data_loader = TextLoader(args.data, args.batch_size, args.seq_length, None)
    vocab_size = data_loader.vocab_size

    config = {
        'model': args.model,
        'seq_length': args.seq_length,
        'batch_size': args.batch_size,
        'vocab_size': vocab_size,
        'embedding_size': args.embedding_size,
        'rnn_size': args.rnn_size,
        'num_layers': args.num_layers
    }

    with open(os.path.join(args.save, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)

    with open(os.path.join(args.save, 'words_vocab.pkl'), 'wb') as f:
        pickle.dump((data_loader.words, data_loader.vocab), f)

    discriminator_scope_name = 'discriminator'
    with tf.variable_scope(discriminator_scope_name):
        embedding_layer = tf.get_variable('embeddings', shape=[vocab_size, args.embedding_size],
                                          initializer=tf.contrib.layers.xavier_initializer(), trainable=False)

    lm_scope_name = 'language_model'
    with tf.variable_scope(lm_scope_name):
        model = LanguageModel(model=config['model'], seq_length=config['seq_length'],
                              batch_size=config['batch_size'], rnn_size=config['rnn_size'],
                              num_layers=config['num_layers'], vocab_size=config['vocab_size'],
                              embedding_layer=embedding_layer, infer=False)

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(model.cost, tvars), args.grad_clip)

    optimizer = tf.train.AdagradOptimizer(args.learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver(tf.global_variables())
    emb_saver = tf.train.Saver([embedding_layer], max_to_keep=1)

    with tf.Session(config=session_config) as session:
        logger.info('Total Parameters: {}'.format(tfutil.count_trainable_parameters()))
        session.run(init_op)

        emb_saver.restore(session, checkpoint_path)

        for epoch_id in range(0, args.num_epochs):
            logger.debug('Epoch: {}'.format(epoch_id))

            data_loader.reset_batch_pointer()
            state = session.run(model.initial_state)

            for batch_id in range(data_loader.pointer, data_loader.num_batches):
                logger.debug('Epoch: {}\tBatch: {}'.format(epoch_id, batch_id))
                x, y = data_loader.next_batch()

                feed_dict = {
                    model.input_data: x,
                    model.targets: y,
                    model.initial_state: state
                }

                train_loss, state, _ = session.run([model.cost, model.final_state, train_op], feed_dict=feed_dict)

                if (epoch_id * data_loader.num_batches + batch_id) % args.batch_size == 0:
                    a = epoch_id * data_loader.num_batches + batch_id
                    b = args.num_epochs * data_loader.num_batches
                    logger.info("{}/{} (epoch {}), train_loss = {:.3f}".format(a, b, epoch_id, train_loss))

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
