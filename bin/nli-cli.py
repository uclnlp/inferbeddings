#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import os
import sys

import numpy as np
import tensorflow as tf

from inferbeddings.io import load_glove, load_word2vec
from inferbeddings.models.training.util import make_batches

from inferbeddings.nli.util import SNLI, count_trainable_parameters, train_tokenizer_on_instances, to_dataset
from inferbeddings.nli import ConditionalBiLSTM, FeedForwardDAM, FeedForwardDAMP, ESIMv1

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
                           choices=['cbilstm', 'ff-dam', 'ff-damp', 'esim1'])
    argparser.add_argument('--optimizer', '-o', action='store', type=str, default='adagrad',
                           choices=['adagrad', 'adam'])

    argparser.add_argument('--embedding-size', action='store', type=int, default=300)
    argparser.add_argument('--representation-size', action='store', type=int, default=200)

    argparser.add_argument('--batch-size', action='store', type=int, default=1024)
    argparser.add_argument('--nb-epochs', action='store', type=int, default=1000)
    argparser.add_argument('--dropout-keep-prob', action='store', type=float, default=1.0)
    argparser.add_argument('--learning-rate', action='store', type=float, default=0.1)
    argparser.add_argument('--clip', '-c', action='store', type=float, default=None)
    argparser.add_argument('--nb-words', action='store', type=int, default=None)
    argparser.add_argument('--seed', action='store', type=int, default=0)

    argparser.add_argument('--semi-sort', action='store_true')
    argparser.add_argument('--fixed-embeddings', '-f', action='store_true')
    argparser.add_argument('--normalized-embeddings', '-n', action='store_true')
    argparser.add_argument('--use-masking', action='store_true')
    argparser.add_argument('--prepend-null-token', action='store_true')

    argparser.add_argument('--save', action='store', type=str, default=None)
    argparser.add_argument('--restore', action='store', type=str, default=None)

    argparser.add_argument('--glove', action='store', type=str, default=None)
    argparser.add_argument('--word2vec', action='store', type=str, default=None)

    argparser.add_argument('--symmetric-contradiction-reg-weight', action='store', type=float, default=None)

    args = argparser.parse_args(argv)

    # Command line arguments
    train_path, valid_path, test_path = args.train, args.valid, args.test

    model_name = args.model
    optimizer_name = args.optimizer

    embedding_size = args.embedding_size
    representation_size = args.representation_size

    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    dropout_keep_prob = args.dropout_keep_prob
    learning_rate = args.learning_rate
    clip_value = args.clip
    nb_words = args.nb_words
    seed = args.seed

    is_semi_sort = args.semi_sort
    is_fixed_embeddings = args.fixed_embeddings
    is_normalized_embeddings = args.normalized_embeddings
    use_masking = args.use_masking
    prepend_null_token = args.prepend_null_token

    save_path = args.save
    restore_path = args.restore

    glove_path = args.glove
    word2vec_path = args.word2vec

    # Experimental RTE regularizers
    symmetric_contradiction_reg_weight = args.symmetric_contradiction_reg_weight

    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    tf.set_random_seed(seed)

    logger.debug('Reading corpus ..')
    train_instances, dev_instances, test_instances = SNLI.generate(
        train_path=train_path, valid_path=valid_path, test_path=test_path)

    logger.info('Train size: {}\tDev size: {}\tTest size: {}'
                .format(len(train_instances), len(dev_instances), len(test_instances)))

    logger.debug('Parsing corpus ..')

    all_instances = train_instances + dev_instances + test_instances
    qs_tokenizer, a_tokenizer = train_tokenizer_on_instances(all_instances, num_words=nb_words)

    contradiction_idx = a_tokenizer.word_index['contradiction'] - 1
    entailment_idx = a_tokenizer.word_index['entailment'] - 1
    neutral_idx = a_tokenizer.word_index['neutral'] - 1

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

    sentence1_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence1')
    sentence2_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence2')

    sentence1_length_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence1_length')
    sentence2_length_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence2_length')

    label_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='label')

    embedding_layer = tf.get_variable('embeddings', shape=[vocab_size, embedding_size],
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      trainable=not is_fixed_embeddings)

    sentence1_embedding = tf.nn.embedding_lookup(embedding_layer, sentence1_ph)
    sentence2_embedding = tf.nn.embedding_lookup(embedding_layer, sentence2_ph)

    dropout_keep_prob_ph = tf.placeholder(tf.float32, name='dropout_keep_prob')

    model_kwargs = dict(
        sequence1=sentence1_embedding, sequence1_length=sentence1_length_ph,
        sequence2=sentence2_embedding, sequence2_length=sentence2_length_ph,
        representation_size=representation_size, dropout_keep_prob=dropout_keep_prob_ph)

    model_class = None
    if model_name == 'cbilstm':
        model_class = ConditionalBiLSTM
    elif model_name == 'ff-dam':
        ff_kwargs = dict(use_masking=use_masking, prepend_null_token=prepend_null_token)
        model_kwargs.update(ff_kwargs)
        model_class = FeedForwardDAM
    elif model_name == 'ff-damp':
        ff_kwargs = dict(use_masking=use_masking, prepend_null_token=prepend_null_token)
        model_kwargs.update(ff_kwargs)
        model_class = FeedForwardDAMP
    elif model_name == 'esim1':
        ff_kwargs = dict(use_masking=use_masking)
        model_kwargs.update(ff_kwargs)
        model_class = ESIMv1

    assert model_class is not None
    model = model_class(**model_kwargs)

    logits = model()
    predictions = tf.argmax(logits, axis=1, name='predictions')

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label_ph)
    loss = tf.reduce_mean(losses)

    if symmetric_contradiction_reg_weight:
        contradiction_prob = tf.nn.softmax(logits)[:, contradiction_idx]

        inv_sequence2, inv_sequence2_length = model_kwargs['sequence1'], model_kwargs['sequence1_length']
        inv_sequence1, inv_sequence1_length = model_kwargs['sequence2'], model_kwargs['sequence2_length']
        inv_model_kwargs = model_kwargs.copy()
        inv_model_kwargs['sequence1'], inv_model_kwargs['sequence1_length'] = inv_sequence1, inv_sequence1_length
        inv_model_kwargs['sequence2'], inv_model_kwargs['sequence2_length'] = inv_sequence2, inv_sequence2_length

        inv_model = model_class(reuse=True, **model_kwargs)
        inv_logits = inv_model()
        inv_contradiction_prob = tf.nn.softmax(inv_logits)[:, contradiction_idx]

        loss += symmetric_contradiction_reg_weight * tf.nn.l2_loss(contradiction_prob - inv_contradiction_prob)

    if clip_value:
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, clip_value)
        training_step = optimizer.apply_gradients(zip(gradients, v))
    else:
        training_step = optimizer.minimize(loss)

    word_idx_ph = tf.placeholder(dtype=tf.int32, name='word_idx')
    word_embedding_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='word_embedding')
    assign_word_embedding = embedding_layer[word_idx_ph, :].assign(word_embedding_ph)

    init_projection_steps = []
    learning_projection_steps = []

    if is_normalized_embeddings:
        unit_sphere_embeddings = constraints.unit_sphere(embedding_layer, norm=1.0)

        init_projection_steps += [unit_sphere_embeddings]
        if not is_fixed_embeddings:
            learning_projection_steps += [unit_sphere_embeddings]

        if prepend_null_token:
            unit_sphere_null_token = constraints.unit_sphere(model.null_token_embedding, norm=1.0)

            init_projection_steps += [unit_sphere_null_token]
            learning_projection_steps += [unit_sphere_null_token]

    predictions_int = tf.cast(predictions, tf.int32)
    labels_int = tf.cast(label_ph, tf.int32)

    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    # session_config.gpu_options.allocator_type = 'BFC'

    with tf.Session(config=session_config) as session:
        logger.debug('Total parameters: {}'.format(count_trainable_parameters()))

        if restore_path:
            saver.restore(session, restore_path)
        else:
            session.run(init_op)

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

            for projection_step in init_projection_steps:
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
                    sentence1_ph: batch_sentences1, sentence2_ph: batch_sentences2,
                    sentence1_length_ph: batch_sizes1, sentence2_length_ph: batch_sizes2,
                    label_ph: batch_labels,
                    dropout_keep_prob_ph: dropout_keep_prob
                }

                _, loss_value = session.run([training_step, loss], feed_dict=batch_feed_dict)

                loss_values += [loss_value]

                for projection_step in learning_projection_steps:
                    session.run([projection_step])

                if (batch_idx > 0 and batch_idx % 1000 == 0) or (batch_start, batch_end) in batches[-1:]:
                    def compute_accuracy(name, dataset):
                        nb_eval_instances = len(dataset['questions'])
                        eval_batches = make_batches(size=nb_eval_instances, batch_size=batch_size)
                        p_vals, l_vals = [], []

                        for batch_start, batch_end in eval_batches:
                            feed_dict = {
                                sentence1_ph: dataset['questions'][batch_start:batch_end],
                                sentence2_ph: dataset['supports'][batch_start:batch_end],
                                sentence1_length_ph: dataset['question_lengths'][batch_start:batch_end],
                                sentence2_length_ph: dataset['support_lengths'][batch_start:batch_end],
                                label_ph: dataset['answers'][batch_start:batch_end],
                                dropout_keep_prob_ph: 1.0
                            }
                            p_val, l_val = session.run([predictions_int, labels_int], feed_dict=feed_dict)
                            p_vals += p_val.tolist()
                            l_vals += l_val.tolist()

                        matches = np.equal(p_vals, l_vals)
                        acc = np.mean(matches)

                        acc_c = np.mean(matches[np.where(np.array(l_vals) == contradiction_idx)])
                        acc_e = np.mean(matches[np.where(np.array(l_vals) == entailment_idx)])
                        acc_n = np.mean(matches[np.where(np.array(l_vals) == neutral_idx)])

                        logger.info('Epoch {0}/Batch {1}\t {2} Accuracy: {3:.4f} - C: {4:.4f}, E: {5:.4f}, N: {6:.4f}'
                                    .format(epoch, batch_idx, name, acc * 100, acc_c * 100, acc_e * 100, acc_n * 100))
                        return acc

                    dev_accuracy = compute_accuracy('Dev', dev_dataset)
                    test_accuracy = compute_accuracy('Test', test_dataset)

                    logger.debug('Epoch {0}/Batch {1}\tAvg loss: {2:.4f}\tDev Accuracy: {3:.2f}\tTest Accuracy: {4:.2f}'
                                 .format(epoch, batch_idx, np.mean(loss_values), dev_accuracy * 100, test_accuracy * 100))

                    if best_dev_accuracy is None or dev_accuracy > best_dev_accuracy:
                        best_dev_accuracy, best_test_accuracy = dev_accuracy, test_accuracy
                        if save_path:
                            ext_save_path = saver.save(session, save_path)
                            logger.info('Model saved in file: {}'.format(ext_save_path))

                    logger.debug('Epoch {0}/Batch {1}\tBest Dev Accuracy: {2:.2f}\tBest Test Accuracy: {3:.2f}'
                                 .format(epoch, batch_idx, best_dev_accuracy * 100, best_test_accuracy * 100))

            def stats(values):
                return '{0:.4f} ± {1:.4f}'.format(round(np.mean(values), 4), round(np.std(values), 4))

            logger.info('Epoch {}\tLoss: {}'.format(epoch, stats(loss_values)))

    logger.info('Training finished.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
