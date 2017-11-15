#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import pickle
import sys
import json

import tensorflow as tf

from inferbeddings.lm.model import LanguageModel

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='data/snli/snli_1.0_train.jsonl.gz')

    parser.add_argument('--vocabulary', type=str, default='models/snli/dam_1/dam_1_index_to_token.p')
    parser.add_argument('--checkpoint', type=str, default='models/snli/dam_1/dam_1')

    parser.add_argument('--nb-words', '-n', type=int, default=200, help='number of words to sample')

    parser.add_argument('--prime', type=str, default=' ', help='prime text')
    parser.add_argument('--pick', type=int, default=1, help='1 = weighted pick, 2 = beam search pick')
    parser.add_argument('--width', type=int, default=4, help='width of the beam search')
    parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces')

    args = parser.parse_args(argv)
    sample(args)


def sample(args):
    vocabulary_path = args.vocabulary
    checkpoint_path = args.checkpoint

    with open(vocabulary_path, 'rb') as f:
        index_to_token = pickle.load(f)

    token_to_index = {token: index for index, token in index_to_token.items()}

    with open(os.path.join(args.save, 'config.json'), 'r') as f:
        config = json.load(f)

    logger.info('Config: {}'.format(str(config)))

    with open(os.path.join(args.save, 'words_vocab.pkl'), 'rb') as f:
        words, vocab = pickle.load(f)

    vocab_size = len(words)

    discriminator_scope_name = 'discriminator'
    with tf.variable_scope(discriminator_scope_name):
        embedding_layer = tf.get_variable('embeddings',
                                          shape=[vocab_size + 3, args.embedding_size],
                                          initializer=tf.contrib.layers.xavier_initializer(),
                                          trainable=False)

    model = LanguageModel(model=config['model'],
                          seq_length=config['seq_length'],
                          batch_size=config['batch_size'],
                          rnn_size=config['rnn_size'],
                          num_layers=config['num_layers'],
                          vocab_size=config['vocab_size'],
                          embedding_layer=embedding_layer,
                          infer=True)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

            sample_value = model.sample(sess, words, vocab, args.nb_words, args.prime, args.sample, args.pick, args.width)
            logger.info('Sample: \"{}\"'.format(sample_value))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
