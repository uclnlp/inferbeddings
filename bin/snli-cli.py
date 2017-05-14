#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import os
import sys

import numpy as np
import tensorflow as tf

from inferbeddings.io import load_glove, load_word2vec
from inferbeddings.models.training.util import make_batches

from inferbeddings.rte import ConditionalBiLSTM
from inferbeddings.rte.dam import SimpleDAM, FeedForwardDAM
from inferbeddings.rte.util import SNLI, count_parameters, train_tokenizer_on_instances, to_dataset, to_feed_dict

from inferbeddings.models.training import constraints

import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Regularising RTE via Adversarial Sets Regularisation',
                                        formatter_class=formatter)

    argparser.add_argument('--train', '-t', action='store', type=str, default='data/snli/snli_1.0_train.jsonl.gz')
    argparser.add_argument('--valid', '-v', action='store', type=str, default='data/snli/snli_1.0_dev.jsonl.gz')
    argparser.add_argument('--test', '-T', action='store', type=str, default='data/snli/snli_1.0_test.jsonl.gz')

    argparser.add_argument('--model', '-m', action='store', type=str, default='cbilstm',
                           choices=['cbilstm', 'simple-dam', 'ff-dam'])
    argparser.add_argument('--optimizer', '-o', action='store', type=str, default='adagrad',
                           choices=['adagrad', 'adam'])

    argparser.add_argument('--embedding-size', action='store', type=int, default=300)
    argparser.add_argument('--representation-size', action='store', type=int, default=200)
    argparser.add_argument('--hidden-size', action='store', type=int, default=200)

    argparser.add_argument('--batch-size', action='store', type=int, default=1024)
    argparser.add_argument('--nb-epochs', action='store', type=int, default=1000)
    argparser.add_argument('--dropout-keep-prob', action='store', type=float, default=1.0)
    argparser.add_argument('--learning-rate', action='store', type=float, default=0.1)
    argparser.add_argument('--seed', action='store', type=int, default=0)

    argparser.add_argument('--semi-sort', action='store_true')
    argparser.add_argument('--fixed-embeddings', '-f', action='store_true')
    argparser.add_argument('--normalized-embeddings', '-n', action='store_true')
    argparser.add_argument('--use-masking', action='store_true')

    argparser.add_argument('--glove', action='store', type=str, default=None)
    argparser.add_argument('--word2vec', action='store', type=str, default=None)

    args = argparser.parse_args(argv)

    train_path, valid_path, test_path = args.train, args.valid, args.test

    model_name = args.model
    optimizer_name = args.optimizer

    embedding_size = args.embedding_size
    representation_size = args.representation_size
    hidden_size = args.hidden_size

    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    dropout_keep_prob = args.dropout_keep_prob
    learning_rate = args.learning_rate
    seed = args.seed

    is_semi_sort = args.semi_sort
    is_fixed_embeddings = args.fixed_embeddings
    is_normalized_embeddings = args.normalized_embeddings
    use_masking = args.use_masking

    glove_path = args.glove
    word2vec_path = args.word2vec

    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    tf.set_random_seed(seed)

    logger.debug('Reading corpus ..')
    train_instances, dev_instances, test_instances = SNLI.generate(
        train_path=train_path, valid_path=valid_path, test_path=test_path)

    logger.info('Train size: {}\tDev size: {}\tTest size: {}'.format(len(train_instances), len(dev_instances), len(test_instances)))

    logger.debug('Parsing corpus ..')

    num_words = None
    qs_tokenizer, a_tokenizer = train_tokenizer_on_instances(train_instances + dev_instances + test_instances, num_words=num_words)

    vocab_size = qs_tokenizer.num_words if qs_tokenizer.num_words else len(qs_tokenizer.word_index) + 1

    max_len = None
    optimizer_class = None
    if optimizer_name == 'adagrad':
        optimizer_class = tf.train.AdagradOptimizer
    elif optimizer_name == 'adam':
        optimizer_class = tf.train.AdamOptimizer
    assert optimizer_class is not None

    optimizer = optimizer_class(learning_rate=learning_rate)

    train_dataset = to_dataset(train_instances, qs_tokenizer, a_tokenizer, max_len=max_len, semi_sort=is_semi_sort)
    dev_dataset = to_dataset(dev_instances, qs_tokenizer, a_tokenizer, max_len=max_len)
    test_dataset = to_dataset(test_instances, qs_tokenizer, a_tokenizer, max_len=max_len)

    questions, supports = train_dataset['questions'], train_dataset['supports']
    question_lengths, support_lengths = train_dataset['question_lengths'], train_dataset['support_lengths']
    answers = train_dataset['answers']

    model_kwargs = dict(
        optimizer=optimizer,
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        l2_lambda=None,
        trainable_embeddings=not is_fixed_embeddings)

    RTEModel = None
    if model_name == 'cbilstm':
        cbilstm_kwargs = dict(hidden_size=hidden_size,
                              dropout_keep_prob=dropout_keep_prob)
        model_kwargs.update(cbilstm_kwargs)
        RTEModel = ConditionalBiLSTM
    elif model_name == 'simple-dam':
        sd_kwargs = dict(use_masking=use_masking)
        model_kwargs.update(sd_kwargs)
        RTEModel = SimpleDAM
    elif model_name == 'ff-dam':
        ff_kwargs = dict(representation_size=representation_size,
                         dropout_keep_prob=dropout_keep_prob,
                         use_masking=use_masking)
        model_kwargs.update(ff_kwargs)
        RTEModel = FeedForwardDAM

    assert RTEModel is not None
    model = RTEModel(**model_kwargs)

    word_idx_ph = tf.placeholder(dtype=tf.int32, name='word_idx')
    word_embedding_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='word_embedding')
    assign_word_embedding = model.embeddings[word_idx_ph, :].assign(word_embedding_ph)

    projection_steps = []
    if is_normalized_embeddings:
        projection_steps += [constraints.unit_sphere(model.embeddings, norm=1.0)]

    correct_predictions = tf.equal(tf.cast(model.predictions, tf.int32), tf.cast(model.label, tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    init_op = tf.global_variables_initializer()

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    with tf.Session(config=session_config) as session:
        session.run(init_op)
        logger.debug('Total parameters: {}'.format(count_parameters()))

        # Initialising pre-trained embeddings
        word_set = {w for w, w_idx in qs_tokenizer.word_index.items() if w_idx < vocab_size}
        word_to_embedding = None
        if glove_path:
            assert os.path.isfile(glove_path)
            word_to_embedding = load_glove(glove_path, word_set)
        elif word2vec_path:
            assert os.path.isfile(word2vec_path)
            word_to_embedding = load_word2vec(word2vec_path, word_set)

        if word_to_embedding:
            logger.info('Initialising the embeddings pre-trained vectors ..')
            for word in word_to_embedding:
                word_idx, word_embedding = qs_tokenizer.word_index[word], word_to_embedding[word]
                assert embedding_size == len(word_embedding)
                session.run(assign_word_embedding, feed_dict={word_idx_ph: word_idx, word_embedding_ph: word_embedding})
            logger.info('Done!')

        for projection_step in projection_steps:
            session.run([projection_step])

        nb_instances = questions.shape[0]
        batches = make_batches(size=nb_instances, batch_size=batch_size)

        best_dev_accuracy, best_test_accuracy = None, None

        for epoch in range(1, nb_epochs + 1):
            order = np.arange(nb_instances) if is_semi_sort else random_state.permutation(nb_instances)

            sentences1, sentences2 = questions[order], supports[order]
            sizes1, sizes2 = question_lengths[order], support_lengths[order]
            labels = answers[order]

            loss_values = []
            for batch_idx, (batch_start, batch_end) in enumerate(batches):
                batch_sentences1, batch_sentences2 = sentences1[batch_start:batch_end], sentences2[batch_start:batch_end]
                batch_sizes1, batch_sizes2 = sizes1[batch_start:batch_end], sizes2[batch_start:batch_end]
                batch_labels = labels[batch_start:batch_end]

                batch_feed_dict = {
                    model.sentence1: batch_sentences1, model.sentence2: batch_sentences2,
                    model.sentence1_size: batch_sizes1, model.sentence2_size: batch_sizes2,
                    model.label: batch_labels}

                _, loss_value = session.run([model.training_step, model.loss], feed_dict=batch_feed_dict)
                loss_values += [loss_value]

                if not is_fixed_embeddings:
                    for projection_step in projection_steps:
                        session.run([projection_step])

                if (batch_idx > 0 and batch_idx % 100 == 0) or (batch_start, batch_end) in batches[-1:]:
                    dev_accuracy = session.run(accuracy, feed_dict=to_feed_dict(model, dev_dataset))
                    test_accuracy = session.run(accuracy, feed_dict=to_feed_dict(model, test_dataset))

                    logger.debug('Epoch {0}/Batch {1}\tAvg loss: {2:.4f}\tDev Accuracy: {3:.2f}\tTest Accuracy: {4:.2f}'
                                 .format(epoch, batch_idx, np.mean(loss_values), dev_accuracy * 100, test_accuracy * 100))
                    if best_dev_accuracy is None or dev_accuracy > best_dev_accuracy:
                        best_dev_accuracy = dev_accuracy
                        best_test_accuracy = test_accuracy

                    logger.debug('Epoch {0}/Batch {1}\tBest Dev Accuracy: {2:.2f}\tBest Test Accuracy: {3:.2f}'
                                 .format(epoch, batch_idx, best_dev_accuracy * 100, best_test_accuracy * 100))

            def stats(values):
                return '{0:.4f} ± {1:.4f}'.format(round(np.mean(values), 4), round(np.std(values), 4))

            logger.info('Epoch {}\tLoss: {}'.format(epoch, stats(loss_values)))

    logger.info('Training finished.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
