#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import os
import sys

import numpy as np
import tensorflow as tf

import tensorflow.contrib.keras as keras

from inferbeddings.models.training.util import make_batches
from inferbeddings.rte import ConditionalBiLSTM
from inferbeddings.rte.util import SNLI, pad_sequences, count_parameters

import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(os.path.basename(sys.argv[0]))


def to_feed_dict(model, dataset):
    return {
        model.sentence1: dataset['questions'], model.sentence2: dataset['supports'],
        model.sentence1_size: dataset['question_lengths'], model.sentence2_size: dataset['support_lengths'],
        model.label: dataset['answers']}


def train_tokenizer_on_instances(instances, num_words=None):
    question_texts = [instance['question'] for instance in instances]
    support_texts = [instance['support'] for instance in instances]
    answer_texts = [instance['answer'] for instance in instances]
    qs_tokenizer, a_tokenizer = keras.preprocessing.text.Tokenizer(num_words=num_words), keras.preprocessing.text.Tokenizer()
    qs_tokenizer.fit_on_texts(question_texts + support_texts)
    a_tokenizer.fit_on_texts(answer_texts)
    return qs_tokenizer, a_tokenizer


def to_dataset(instances, qs_tokenizer, a_tokenizer, max_len=None):
    question_texts = [instance['question'] for instance in instances]
    support_texts = [instance['support'] for instance in instances]
    answer_texts = [instance['answer'] for instance in instances]

    assert qs_tokenizer is not None and a_tokenizer is not None

    questions = qs_tokenizer.texts_to_sequences(question_texts)
    question_lenths = [len(q) for q in questions]

    supports = [[s] for s in qs_tokenizer.texts_to_sequences(support_texts)]
    support_lenghs = [[len(s)] for [s] in supports]

    answers = [answers - 1 for [answers] in a_tokenizer.texts_to_sequences(answer_texts)]

    assert set(answers) == {0, 1, 2}

    return {
        'questions': pad_sequences(questions, max_len=max_len),
        'supports': pad_sequences([s for [s] in supports], max_len=max_len),
        'question_lengths': np.clip(a=np.array(question_lenths), a_min=0, a_max=max_len),
        'support_lengths': np.clip(a=np.array(support_lenghs)[:, 0], a_min=0, a_max=max_len),
        'answers': np.array(answers)}


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Regularising RTE via Adversarial Sets Regularisation', formatter_class=formatter)

    argparser.add_argument('--train', action='store', type=str, default='data/snli/snli_1.0_train.jsonl.gz')
    argparser.add_argument('--valid', action='store', type=str, default='data/snli/snli_1.0_dev.jsonl.gz')
    argparser.add_argument('--test', action='store', type=str, default='data/snli/snli_1.0_test.jsonl.gz')

    argparser.add_argument('--embedding-size', action='store', type=int, default=300)
    argparser.add_argument('--batch-size', action='store', type=int, default=1024)
    argparser.add_argument('--num-units', action='store', type=int, default=300)
    argparser.add_argument('--nb-epochs', action='store', type=int, default=1000)
    argparser.add_argument('--dropout-keep-prob', action='store', type=int, default=1.0)
    argparser.add_argument('--learning-rate', action='store', type=int, default=0.001)

    args = argparser.parse_args(argv)

    train_path, valid_path, test_path = args.train, args.valid, args.test

    embedding_size = args.embedding_size
    batch_size = args.batch_size
    num_units = args.num_units
    nb_epochs = args.nb_epochs
    dropout_keep_prob = args.dropout_keep_prob
    learning_rate = args.learning_rate

    logger.debug('Reading corpus ..')
    train_instances, dev_instances, test_instances = SNLI.generate(
        train_path=train_path, valid_path=valid_path, test_path=test_path)

    train_instances = dev_instances = test_instances = train_instances[:100]
    logger.info('Train size: {}\tDev size: {}\tTest size: {}'.format(len(train_instances), len(dev_instances), len(test_instances)))

    logger.debug('Parsing corpus ..')

    num_words = None
    qs_tokenizer, a_tokenizer = train_tokenizer_on_instances(train_instances + dev_instances + test_instances, num_words=num_words)

    vocab_size = qs_tokenizer.num_words if qs_tokenizer.num_words else len(qs_tokenizer.word_index) + 1

    max_len = None
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    train_dataset = to_dataset(train_instances, qs_tokenizer, a_tokenizer, max_len=max_len)
    dev_dataset = to_dataset(dev_instances, qs_tokenizer, a_tokenizer, max_len=max_len)
    test_dataset = to_dataset(test_instances, qs_tokenizer, a_tokenizer, max_len=max_len)

    questions, supports = train_dataset['questions'], train_dataset['supports']
    question_lengths, support_lengths = train_dataset['question_lengths'], train_dataset['support_lengths']
    answers = train_dataset['answers']

    model = ConditionalBiLSTM(optimizer=optimizer, num_units=num_units, num_classes=3,
                              vocab_size=vocab_size, embedding_size=embedding_size,
                              dropout_keep_prob=dropout_keep_prob, l2_lambda=1e-5)

    word_idx_ph = tf.placeholder(dtype=tf.int32, name='word_idx')
    word_embedding_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='word_embedding')
    assign_word_embedding = model.embeddings[word_idx_ph, :].assign(word_embedding_ph)

    correct_predictions = tf.equal(tf.cast(model.predictions, tf.int32), tf.cast(model.label, tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    init_op = tf.global_variables_initializer()

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    with tf.Session(config=session_config) as session:
        session.run(init_op)
        logger.debug('Total parameters: {}'.format(count_parameters()))

        glove_path = os.path.expanduser('~/data/glove/glove.840B.300d.txt')
        if embedding_size == 300 and os.path.isfile(glove_path):
            from derte.io.embeddings import load_glove
            logger.info('Initialising the embeddings with GloVe vectors ..')

            word_set = {w for w, w_idx in qs_tokenizer.word_index.items() if w_idx < vocab_size}
            with open(glove_path, 'r') as stream:
                word_to_embedding = load_glove(stream=stream, words=word_set)

            for word in word_to_embedding:
                word_idx, word_embedding = qs_tokenizer.word_index[word], word_to_embedding[word]
                session.run(assign_word_embedding, feed_dict={word_idx_ph: word_idx, word_embedding_ph: word_embedding})

            logger.info('Done!')

        random_state = np.random.RandomState(0)

        nb_instances = questions.shape[0]
        batches = make_batches(size=nb_instances, batch_size=batch_size)

        for epoch in range(1, nb_epochs + 1):
            order = random_state.permutation(nb_instances)

            sentences1, sentences2 = questions[order], supports[order]
            sizes1, sizes2 = question_lengths[order], support_lengths[order]
            labels = answers[order]

            loss_values, correct_predictions_values = [], []

            for i, (batch_start, batch_end) in enumerate(batches):
                batch_sentences1, batch_sentences2 = sentences1[batch_start:batch_end], sentences2[batch_start:batch_end]
                batch_sizes1, batch_sizes2 = sizes1[batch_start:batch_end], sizes2[batch_start:batch_end]
                batch_labels = labels[batch_start:batch_end]

                batch_feed_dict = {
                    model.sentence1: batch_sentences1, model.sentence2: batch_sentences2,
                    model.sentence1_size: batch_sizes1, model.sentence2_size: batch_sizes2,
                    model.label: batch_labels}

                _, loss_value, correct_predictions_value =\
                    session.run([model.training_step, model.loss, correct_predictions], feed_dict=batch_feed_dict)

                loss_values += loss_value.tolist()
                correct_predictions_values += correct_predictions_value.tolist()

                if (i > 0 and i % 100 == 0) or (batch_start, batch_end) in batches[-1:]:
                    train_accuracy = np.mean(correct_predictions_values)
                    dev_accuracy = session.run(accuracy, feed_dict=to_feed_dict(model, dev_dataset))
                    test_accuracy = session.run(accuracy, feed_dict=to_feed_dict(model, test_dataset))

                    logger.debug('Epoch {0}/{1}\tTrain Accuracy: {2:.2f}\tDev Accuracy: {3:.2f}\tTest Accuracy: {4:.2f}'
                                 .format(epoch, i, train_accuracy * 100, dev_accuracy * 100, test_accuracy * 100))

            def stats(values):
                return '{0:.4f} ± {1:.4f}'.format(round(np.mean(values), 4), round(np.std(values), 4))

            logger.info('Epoch {}\tLoss: {}'.format(epoch, stats(loss_values)))

    logger.info('Training finished.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
