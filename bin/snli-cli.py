#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import os
import sys

import numpy as np
import tensorflow as tf

import tensorflow.contrib.keras as keras

from inferbeddings.io import load_glove, load_word2vec
from inferbeddings.models.training.util import make_batches

from inferbeddings.rte import ConditionalBiLSTM
from inferbeddings.rte.dam import SimpleDAM, FeedForwardDAM
from inferbeddings.rte.util import SNLI, pad_sequences, count_parameters

from inferbeddings.models.training import constraints

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


def to_dataset(instances, qs_tokenizer, a_tokenizer, max_len=None, semi_sort=False):
    question_texts = [instance['question'] for instance in instances]
    support_texts = [instance['support'] for instance in instances]
    answer_texts = [instance['answer'] for instance in instances]

    assert qs_tokenizer is not None and a_tokenizer is not None

    questions = qs_tokenizer.texts_to_sequences(question_texts)
    supports = [s for s in qs_tokenizer.texts_to_sequences(support_texts)]
    answers = [answers - 1 for [answers] in a_tokenizer.texts_to_sequences(answer_texts)]

    """
    <<For efficient batching in TensorFlow, we semi-sorted the training data to first contain examples
    where both sentences had length less than 20, followed by those with length less than 50, and
    then the rest. This ensured that most training batches contained examples of similar length.>>

    -- https://arxiv.org/pdf/1606.01933.pdf
    """
    if semi_sort:
        triples_under_20, triples_under_50, triples_under_nfty = [], [], []
        for q, s, a in zip(questions, supports, answers):
            if len(q) < 20 and len(s) < 20:
                triples_under_20 += [(q, s, a)]
            elif len(q) < 50 and len(s) < 50:
                triples_under_50 += [(q, s, a)]
            else:
                triples_under_nfty += [(q, s, a)]
        questions, supports, answers = [], [], []
        for q, s, a in triples_under_20 + triples_under_50 + triples_under_nfty:
            questions += [q]
            supports += [s]
            answers += [a]

    question_lenths = [len(q) for q in questions]
    support_lenghs = [len(s) for s in supports]

    assert set(answers) == {0, 1, 2}

    return {
        'questions': pad_sequences(questions, max_len=max_len),
        'supports': pad_sequences(supports, max_len=max_len),
        'question_lengths': np.clip(a=np.array(question_lenths), a_min=0, a_max=max_len),
        'support_lengths': np.clip(a=np.array(support_lenghs), a_min=0, a_max=max_len),
        'answers': np.array(answers)}


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

    argparser.add_argument('--embedding-size', action='store', type=int, default=300)
    argparser.add_argument('--representation-size', action='store', type=int, default=200)

    argparser.add_argument('--batch-size', action='store', type=int, default=1024)
    argparser.add_argument('--nb-epochs', action='store', type=int, default=1000)
    argparser.add_argument('--dropout-keep-prob', action='store', type=float, default=1.0)
    argparser.add_argument('--learning-rate', action='store', type=float, default=0.001)
    argparser.add_argument('--seed', action='store', type=int, default=0)

    argparser.add_argument('--semi-sort', action='store_true')
    argparser.add_argument('--fixed-embeddings', '-f', action='store_true')
    argparser.add_argument('--normalized-embeddings', '-n', action='store_true')

    argparser.add_argument('--glove', action='store', type=str, default=None)
    argparser.add_argument('--word2vec', action='store', type=str, default=None)

    args = argparser.parse_args(argv)

    train_path, valid_path, test_path = args.train, args.valid, args.test

    model_name = args.model
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    representation_size = args.representation_size

    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    dropout_keep_prob = args.dropout_keep_prob
    learning_rate = args.learning_rate
    seed = args.seed

    is_semi_sort = args.semi_sort
    is_fixed_embeddings = args.fixed_embeddings
    is_normalized_embeddings = args.normalized_embeddings

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
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

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
        l2_lambda=1e-5,
        trainable_embeddings=not is_fixed_embeddings)

    RTEModel = None
    if model_name == 'cbilstm':
        cbilstm_kwargs = dict(hidden_size=hidden_size, dropout_keep_prob=dropout_keep_prob)
        model_kwargs.update(cbilstm_kwargs)
        RTEModel = ConditionalBiLSTM
    elif model_name == 'simple-dam':
        RTEModel = SimpleDAM
    if model_name == 'ff-dam':
        ff_kwargs = dict(representation_size=representation_size)
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

        for epoch in range(1, nb_epochs + 1):
            order = np.arange(nb_instances) if is_semi_sort else random_state.permutation(nb_instances)

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

                for projection_step in projection_steps:
                    session.run([projection_step])

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
