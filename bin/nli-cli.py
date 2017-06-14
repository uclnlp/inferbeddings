#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import os
import sys

import numpy as np
import tensorflow as tf

from inferbeddings.io import load_glove, load_word2vec
from inferbeddings.models.training.util import make_batches

from inferbeddings.nli import util, tfutil
from inferbeddings.nli import ConditionalBiLSTM, FeedForwardDAM, FeedForwardDAMP, ESIMv1

from inferbeddings.nli.regularizers.base import symmetry_contradiction_regularizer
from inferbeddings.nli.regularizers.adversarial import AdversarialSets

from inferbeddings.models.training import constraints

from inferbeddings.nli.evaluation import accuracy

import logging

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

    argparser.add_argument('--nb-epochs', '-e', action='store', type=int, default=1000)
    argparser.add_argument('--nb-discriminator-epochs', '-D', action='store', type=int, default=1)
    argparser.add_argument('--nb-adversary-epochs', '-A', action='store', type=int, default=1000)

    argparser.add_argument('--dropout-keep-prob', action='store', type=float, default=1.0)
    argparser.add_argument('--learning-rate', action='store', type=float, default=0.1)
    argparser.add_argument('--clip', '-c', action='store', type=float, default=None)
    argparser.add_argument('--nb-words', action='store', type=int, default=None)
    argparser.add_argument('--seed', action='store', type=int, default=0)

    argparser.add_argument('--semi-sort', action='store_true')
    argparser.add_argument('--fixed-embeddings', '-f', action='store_true')
    argparser.add_argument('--normalized-embeddings', '-n', action='store_true')
    argparser.add_argument('--use-masking', action='store_true')

    argparser.add_argument('--save', action='store', type=str, default=None)
    argparser.add_argument('--restore', action='store', type=str, default=None)

    argparser.add_argument('--glove', action='store', type=str, default=None)
    argparser.add_argument('--word2vec', action='store', type=str, default=None)

    argparser.add_argument('--rule0-weight', '-0', action='store', type=float, default=None)
    argparser.add_argument('--rule1-weight', '-1', action='store', type=float, default=None)
    argparser.add_argument('--rule2-weight', '-2', action='store', type=float, default=None)

    args = argparser.parse_args(argv)

    # Command line arguments
    train_path, valid_path, test_path = args.train, args.valid, args.test

    model_name = args.model
    optimizer_name = args.optimizer

    embedding_size = args.embedding_size
    representation_size = args.representation_size

    batch_size = args.batch_size

    nb_epochs = args.nb_epochs
    nb_discriminator_epochs = args.nb_discriminator_epochs
    nb_adversary_epochs = args.nb_adversary_epochs

    dropout_keep_prob = args.dropout_keep_prob
    learning_rate = args.learning_rate
    clip_value = args.clip
    nb_words = args.nb_words
    seed = args.seed

    is_semi_sort = args.semi_sort
    is_fixed_embeddings = args.fixed_embeddings
    is_normalized_embeddings = args.normalized_embeddings
    use_masking = args.use_masking

    save_path = args.save
    restore_path = args.restore

    glove_path = args.glove
    word2vec_path = args.word2vec

    # Experimental RTE regularizers
    rule0_weight = args.rule0_weight
    rule1_weight = args.rule1_weight
    rule2_weight = args.rule2_weight

    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    tf.set_random_seed(seed)

    logger.debug('Reading corpus ..')
    train_instances, dev_instances, test_instances = util.SNLI.generate(
        train_path=train_path, valid_path=valid_path, test_path=test_path)

    logger.info('Train size: {}\tDev size: {}\tTest size: {}'
                .format(len(train_instances), len(dev_instances), len(test_instances)))

    logger.debug('Parsing corpus ..')
    all_instances = train_instances + dev_instances + test_instances
    qs_tokenizer, a_tokenizer = util.train_tokenizer_on_instances(all_instances, num_words=nb_words)

    # Indices (in the final logits) corresponding to the three NLI classes
    contradiction_idx = a_tokenizer.word_index['contradiction'] - 1
    entailment_idx = a_tokenizer.word_index['entailment'] - 1
    neutral_idx = a_tokenizer.word_index['neutral'] - 1

    # Size of the vocabulary (number of embedding vectors)
    vocab_size = qs_tokenizer.num_words if qs_tokenizer.num_words else max(qs_tokenizer.word_index.values()) + 1

    max_len = None
    optimizer_name_to_class = {
        'adagrad': tf.train.AdagradOptimizer,
        'adam': tf.train.AdamOptimizer
    }
    optimizer_class = optimizer_name_to_class[optimizer_name]
    assert optimizer_class

    optimizer = optimizer_class(learning_rate=learning_rate)

    train_dataset = util.to_dataset(train_instances, qs_tokenizer, a_tokenizer, max_len=max_len,
                                    semi_sort=is_semi_sort)
    dev_dataset = util.to_dataset(dev_instances, qs_tokenizer, a_tokenizer, max_len=max_len)
    test_dataset = util.to_dataset(test_instances, qs_tokenizer, a_tokenizer, max_len=max_len)

    questions, supports = train_dataset['questions'], train_dataset['supports']
    question_lengths, support_lengths = train_dataset['question_lengths'], train_dataset['support_lengths']
    answers = train_dataset['answers']

    sentence1_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence1')
    sentence2_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence2')

    sentence1_len_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence1_length')
    sentence2_len_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence2_length')

    def clip_sentence(sentence, sizes):
        """
        Clip the input sentence placeholders to the length of the longest one in the batch.
        This saves processing time.

        :param sentence: tensor with shape (batch, time_steps)
        :param sizes: tensor with shape (batch)
        :return: tensor with shape (batch, time_steps)
        """
        return tf.slice(sentence, [0, 0], tf.stack([-1, tf.reduce_max(sizes)]))

    clipped_sentence1 = clip_sentence(sentence1_ph, sentence1_len_ph)
    clipped_sentence2 = clip_sentence(sentence2_ph, sentence2_len_ph)

    label_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='label')

    discriminator_scope_name = 'discriminator'
    with tf.variable_scope(discriminator_scope_name):
        embedding_layer = tf.get_variable('embeddings',
                                          shape=[vocab_size, embedding_size],
                                          initializer=tf.contrib.layers.xavier_initializer(),
                                          trainable=not is_fixed_embeddings)

        sentence1_embedding = tf.nn.embedding_lookup(embedding_layer, clipped_sentence1)
        sentence2_embedding = tf.nn.embedding_lookup(embedding_layer, clipped_sentence2)

        dropout_keep_prob_ph = tf.placeholder(tf.float32, name='dropout_keep_prob')

        model_kwargs = dict(
            sequence1=sentence1_embedding, sequence1_length=sentence1_len_ph,
            sequence2=sentence2_embedding, sequence2_length=sentence2_len_ph,
            representation_size=representation_size, dropout_keep_prob=dropout_keep_prob_ph)

        mode_name_to_class = {
            'cbilstm': ConditionalBiLSTM,
            'ff-dam': FeedForwardDAM,
            'ff-damp': FeedForwardDAMP,
            'esim1': ESIMv1
        }

        if model_name in {'ff-dam', 'ff-damp', 'esim1'}:
            model_kwargs.update(dict(use_masking=use_masking))

        model_class = mode_name_to_class[model_name]

        assert model_class is not None
        model = model_class(**model_kwargs)

        logits = model()
        predictions = tf.argmax(logits, axis=1, name='predictions')

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label_ph)
        loss = tf.reduce_mean(losses)

        if rule0_weight:
            loss += rule0_weight * symmetry_contradiction_regularizer(model_class, model_kwargs,
                                                                      contradiction_idx=contradiction_idx)

    discriminator_vars = tfutil.get_variables_in_scope(discriminator_scope_name)
    discriminator_init_op = tf.variables_initializer(discriminator_vars)

    discriminator_optimizer_scope_name = 'discriminator_optimizer'
    with tf.variable_scope(discriminator_optimizer_scope_name):
        if clip_value:
            gradients, v = zip(*optimizer.compute_gradients(loss, var_list=discriminator_vars))
            gradients, _ = tf.clip_by_global_norm(gradients, clip_value)
            training_step = optimizer.apply_gradients(zip(gradients, v))
        else:
            training_step = optimizer.minimize(loss, var_list=discriminator_vars)

    discriminator_optimizer_vars = tfutil.get_variables_in_scope(discriminator_optimizer_scope_name)
    discriminator_optimizer_init_op = tf.variables_initializer(discriminator_optimizer_vars)

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

    predictions_int = tf.cast(predictions, tf.int32)
    labels_int = tf.cast(label_ph, tf.int32)

    use_adversarial_training = rule1_weight or rule2_weight

    if use_adversarial_training:
        adversary_scope_name = discriminator_scope_name
        with tf.variable_scope(adversary_scope_name):
            adversarial = AdversarialSets(model_class=model_class, model_kwargs=model_kwargs, embedding_size=embedding_size,
                                          scope_name='adversary', batch_size=32, sequence_length=10,
                                          entailment_idx=entailment_idx, contradiction_idx=contradiction_idx, neutral_idx=neutral_idx)

            adversary_loss = tf.constant(0.0, dtype=tf.float32)
            adversary_vars = []

            if rule1_weight:
                rule1_loss, rule1_vars = adversarial.rule1()
                adversary_loss += rule1_weight * tf.reduce_max(rule1_loss)
                adversary_vars += rule1_vars
            if rule2_weight:
                rule2_loss, rule2_vars = adversarial.rule2()
                adversary_loss += rule2_weight * tf.reduce_max(rule2_loss)
                adversary_vars += rule2_vars

        adversary_init_op = tf.variables_initializer(adversary_vars)

        adv_opt_scope_name = 'adversary_optimizer'
        with tf.variable_scope(adv_opt_scope_name):
            adversary_optimizer = optimizer_class(learning_rate=learning_rate)
            adversary_training_step = adversary_optimizer.minimize(- adversary_loss, var_list=adversary_vars)

            adversary_optimizer_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=adv_opt_scope_name)
            adversary_optimizer_init_op = tf.variables_initializer(adversary_optimizer_vars)

        adversary_projection_steps = []
        for var in adversary_vars:
            if is_normalized_embeddings:
                unit_sphere_adversarial_embeddings = constraints.unit_sphere(var, norm=1.0, axis=-1)
                adversary_projection_steps += [unit_sphere_adversarial_embeddings]

            adversarial_batch_size, sentence_len = var.get_shape()[0].value, var.get_shape()[1].value

            def token_init_op(_var, token_idx, target_idx):
                token_emb = tf.nn.embedding_lookup(embedding_layer, token_idx)
                tiled_token_emb = tf.tile(tf.expand_dims(token_emb, 0), (adversarial_batch_size, 1))
                init_token_op = _var[:, target_idx, :].assign(tiled_token_emb),
                return init_token_op

            adversary_projection_steps += [
                token_init_op(var, qs_tokenizer.bos_idx, 0),
                token_init_op(var, qs_tokenizer.eos_idx, sentence_len - 1)]

    saver = tf.train.Saver(discriminator_vars + discriminator_optimizer_vars)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    with tf.Session(config=session_config) as session:
        logger.debug('Total parameters: {}'.format(tfutil.count_trainable_parameters()))

        if use_adversarial_training:
            session.run([adversary_init_op, adversary_optimizer_init_op])

        if restore_path:
            saver.restore(session, restore_path)
        else:
            session.run([discriminator_init_op, discriminator_optimizer_init_op])

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

        best_dev_acc, best_test_acc = None, None

        for epoch in range(1, nb_epochs + 1):
            for d_epoch in range(1, nb_discriminator_epochs + 1):
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
                        sentence1_len_ph: batch_sizes1, sentence2_len_ph: batch_sizes2,
                        label_ph: batch_labels, dropout_keep_prob_ph: dropout_keep_prob
                    }

                    _, loss_value = session.run([training_step, loss], feed_dict=batch_feed_dict)

                    loss_values += [loss_value]

                    for projection_step in learning_projection_steps:
                        session.run([projection_step])

                    if (batch_idx > 0 and batch_idx % 1000 == 0) or (batch_start, batch_end) in batches[-1:]:
                        dev_acc, _, _, _ = accuracy(session, dev_dataset, 'Dev',
                                                    sentence1_ph, sentence1_len_ph, sentence2_ph, sentence2_len_ph,
                                                    label_ph, dropout_keep_prob_ph, predictions_int, labels_int,
                                                    contradiction_idx, entailment_idx, neutral_idx, batch_size)

                        test_acc, _, _, _ = accuracy(session, test_dataset, 'Test',
                                                     sentence1_ph, sentence1_len_ph, sentence2_ph, sentence2_len_ph,
                                                     label_ph, dropout_keep_prob_ph, predictions_int, labels_int,
                                                     contradiction_idx, entailment_idx, neutral_idx, batch_size)

                        logger.debug('Epoch {0}/{1}/{2}\tAvg Loss: {3:.4f}\tDev Acc: {4:.2f}\tTest Acc: {5:.2f}'
                                     .format(epoch, d_epoch, batch_idx, np.mean(loss_values),
                                             dev_acc * 100, test_acc * 100))

                        if best_dev_acc is None or dev_acc > best_dev_acc:
                            best_dev_acc, best_test_acc = dev_acc, test_acc
                            if save_path:
                                logger.info('Model saved in file: {}'.format(saver.save(session, save_path)))

                        logger.debug('Epoch {0}/{1}/{2}\tBest Dev Accuracy: {3:.2f}\tBest Test Accuracy: {4:.2f}'
                                     .format(epoch, d_epoch, batch_idx, best_dev_acc * 100, best_test_acc * 100))

                def stats(values):
                    return '{0:.4f} ± {1:.4f}'.format(round(np.mean(values), 4), round(np.std(values), 4))
                logger.info('Epoch {}\tLoss: {}'.format(epoch, stats(loss_values)))

            if use_adversarial_training:
                session.run([adversary_init_op, adversary_optimizer_init_op])

                for a_epoch in range(1, nb_adversary_epochs + 1):
                    adversary_feed_dict = {
                        dropout_keep_prob_ph: 1.0
                    }
                    _, adversarial_loss_value = session.run([adversary_training_step, adversary_loss],
                                                            feed_dict=adversary_feed_dict)

                    logger.debug('Adversary Epoch {0}/{1}\tLoss: {2}'.format(epoch, a_epoch, adversarial_loss_value))

                    for projection_step in adversary_projection_steps:
                        session.run(projection_step)

    logger.info('Training finished.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
