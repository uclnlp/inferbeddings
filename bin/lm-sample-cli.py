#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import pickle
import sys

import tensorflow as tf

from inferbeddings.lm.legacy.model import LanguageModel

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='save',
                        help='model directory to load stored checkpointed models from')
    parser.add_argument('-n', type=int, default=200, help='number of words to sample')
    parser.add_argument('--prime', type=str, default=' ', help='prime text')
    parser.add_argument('--pick', type=int, default=1, help='1 = weighted pick, 2 = beam search pick')
    parser.add_argument('--width', type=int, default=4, help='width of the beam search')
    parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces')

    args = parser.parse_args(argv)
    sample(args)


def sample(args):
    with open(os.path.join(args.save, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)

    with open(os.path.join(args.save, 'words_vocab.pkl'), 'rb') as f:
        words, vocab = pickle.load(f)

    vocab_size = len(words)

    embedding_layer = tf.get_variable('embeddings', shape=[vocab_size, args.embedding_size],
                                      initializer=tf.contrib.layers.xavier_initializer(), trainable=False)

    model = LanguageModel(model=config['model'], seq_length=config['seq_length'], batch_size=config['batch_size'],
                          rnn_size=config['rnn_size'], num_layers=config['num_layers'], vocab_size=config['vocab_size'],
                          embedding_layer=embedding_layer, infer=True)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(model.sample(sess, words, vocab, args.n, args.prime, args.sample, args.pick, args.width))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
