#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse
import time

import pickle

import tensorflow as tf

from inferbeddings.lm.loader import TextLoader
from inferbeddings.lm.model import LanguageModel

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/lm/neuromancer', help='data directory containing input.txt')
    parser.add_argument('--input_encoding', type=str, default=None,
                        help='character encoding of input.txt, from https://docs.python.org/3/library/codecs.html#standard-encodings')
    parser.add_argument('--save_dir', type=str, default='save', help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=256, help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='rnn', help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=50, help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=25, help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000, help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5., help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97, help='decay rate for rmsprop')
    parser.add_argument('--init_from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                             'config.pkl'        : configuration;
                             'words_vocab.pkl'   : vocabulary definitions;
                             'checkpoint'        : paths to model file(s) (created by tf).
                                                   Note: this file contains absolute paths, be careful when moving files around;
                             'model.ckpt-*'      : file(s) with model definition (created by tf)
                             """)
    args = parser.parse_args(argv)
    train(args)


def train(args):
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length, args.input_encoding)
    vocab_size = data_loader.vocab_size

    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from)
        assert os.path.isfile(os.path.join(args.init_from, "config.pkl"))
        assert os.path.isfile(os.path.join(args.init_from, "words_vocab.pkl"))

        ckpt = tf.train.get_checkpoint_state(args.init_from)

        assert ckpt, "No checkpoint found"
        assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = pickle.load(f)

        need_be_same = ["model", "rnn_size", "num_layers", "seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme] == vars(args)[checkme], "Disagreement on '{}'".format(checkme)

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'words_vocab.pkl'), 'rb') as f:
            saved_words, saved_vocab = pickle.load(f)

        assert saved_words == data_loader.words, "Data and loaded model disagree on word set!"
        assert saved_vocab == data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"

    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'wb') as f:
        pickle.dump((data_loader.words, data_loader.vocab), f)

    model = LanguageModel(
        model=args.model, seq_length=args.seq_length, batch_size=args.batch_size, rnn_size=args.rnn_size,
        num_layers=args.num_layers, learning_rate=args.learning_rate, grad_clip=args.grad_clip,
        vocab_size=vocab_size, infer=False)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    with tf.Session(config=session_config) as session:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())

        # restore model
        if args.init_from is not None:
            saver.restore(session, ckpt.model_checkpoint_path)

        for epoch_id in range(0, args.num_epochs):
            logger.debug('Epoch: {}'.format(epoch_id))

            data_loader.reset_batch_pointer()
            state = session.run(model.initial_state)

            for batch_id in range(data_loader.pointer, data_loader.num_batches):
                logger.debug('Epoch: {}\tBatch: {}'.format(epoch_id, batch_id))
                x, y = data_loader.next_batch()

                feed = {model.input_data: x, model.targets: y, model.initial_state: state}
                train_loss, state, _ = session.run([model.cost, model.final_state, model.train_op], feed)

                if (epoch_id * data_loader.num_batches + batch_id) % args.batch_size == 0:
                    logger.info("{}/{} (epoch {}), train_loss = {:.3f}"
                                .format(epoch_id * data_loader.num_batches + batch_id, args.num_epochs * data_loader.num_batches,
                                        epoch_id, train_loss))

                if (epoch_id * data_loader.num_batches + batch_id) % args.save_every == 0 or \
                        (epoch_id == args.num_epochs - 1 and batch_id == data_loader.num_batches-1):

                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(session, checkpoint_path, global_step=epoch_id * data_loader.num_batches + batch_id)
                    logger.info("model saved to {}".format(checkpoint_path))

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
