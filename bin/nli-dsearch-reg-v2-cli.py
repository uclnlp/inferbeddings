#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sample usage:

$ python3 ./bin/nli-dsearch-reg-v2-cli.py -f -n -m ff-dam --batch-size 32 --dropout-keep-prob 0.8
--representation-size 200 --optimizer adagrad --learning-rate 0.05 -c 100 -i uniform --nb-epochs 10 --has-bos --has-unk
-p -S --restore models/snli/dam_1/dam_1 --04 0.0 -P sum -E data/snli/generated/snli_1.0_contradictions_*.gz
--hard-save models/snli/dam_1/acl/batch_dsearch_reg_v1/dam_1_0
"""


import argparse

import os
import sys

import pickle
import socket
import copy

import numpy as np
import tensorflow as tf

from inferbeddings.io import load_glove, load_glove_words
from inferbeddings.models.training.util import make_batches

from inferbeddings.nli import util, tfutil
from inferbeddings.nli.evaluation import util as eutil
from inferbeddings.nli import ConditionalBiLSTM, FeedForwardDAM, FeedForwardDAMP, FeedForwardDAMS, ESIMv1

import inferbeddings.nli.regularizers.base as R

from inferbeddings.models.training import constraints

from inferbeddings.nli.generate.generator import Generator
from inferbeddings.nli.generate.scorer import LMScorer
from inferbeddings.nli.generate.scorer import IScorer

from inferbeddings.nli.evaluation import accuracy, stats

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    logger.info('Command line: {}'.format(' '.join(arg for arg in argv)))

    def fmt(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Regularising RTE via Adversarial Sets Regularisation', formatter_class=fmt)

    argparser.add_argument('--train', '-t', action='store', type=str, default='data/snli/snli_1.0_train.jsonl.gz')
    argparser.add_argument('--valid', '-v', action='store', type=str, default='data/snli/snli_1.0_dev.jsonl.gz')
    argparser.add_argument('--test', '-T', action='store', type=str, default='data/snli/snli_1.0_test.jsonl.gz')

    argparser.add_argument('--model', '-m', action='store', type=str, default='ff-dam',
                           choices=['cbilstm', 'ff-dam', 'ff-damp', 'ff-dams', 'esim1'])

    argparser.add_argument('--optimizer', '-o', action='store', type=str, default='adagrad',
                           choices=['adagrad', 'adam'])

    argparser.add_argument('--embedding-size', action='store', type=int, default=300)
    argparser.add_argument('--representation-size', action='store', type=int, default=200)

    argparser.add_argument('--batch-size', action='store', type=int, default=32)

    argparser.add_argument('--nb-epochs', '-e', action='store', type=int, default=1000)
    argparser.add_argument('--nb-discriminator-epochs', '-D', action='store', type=int, default=1)
    argparser.add_argument('--nb-adversary-epochs', '-A', action='store', type=int, default=1000)

    argparser.add_argument('--dropout-keep-prob', action='store', type=float, default=1.0)
    argparser.add_argument('--learning-rate', action='store', type=float, default=0.1)
    argparser.add_argument('--clip', '-c', action='store', type=float, default=None)
    argparser.add_argument('--nb-words', action='store', type=int, default=None)
    argparser.add_argument('--seed', action='store', type=int, default=0)
    argparser.add_argument('--std-dev', action='store', type=float, default=0.01)

    argparser.add_argument('--has-bos', action='store_true', default=False, help='Has <Beginning Of Sentence> token')
    argparser.add_argument('--has-eos', action='store_true', default=False, help='Has <End Of Sentence> token')
    argparser.add_argument('--has-unk', action='store_true', default=False, help='Has <Unknown Word> token')
    argparser.add_argument('--lower', '-l', action='store_true', default=False, help='Lowercase the corpus')

    argparser.add_argument('--initialize-embeddings', '-i', action='store', type=str, default=None,
                           choices=['normal', 'uniform'])

    argparser.add_argument('--fixed-embeddings', '-f', action='store_true')
    argparser.add_argument('--normalize-embeddings', '-n', action='store_true')
    argparser.add_argument('--only-use-pretrained-embeddings', '-p', action='store_true',
                           help='Only use pre-trained word embeddings')
    argparser.add_argument('--semi-sort', '-S', action='store_true')

    argparser.add_argument('--save', action='store', type=str, default=None)
    argparser.add_argument('--hard-save', action='store', type=str, default=None)
    argparser.add_argument('--restore', action='store', type=str, default=None)

    argparser.add_argument('--glove', action='store', type=str, default=None)

    for i in range(0, 13):
        argparser.add_argument('--rule{:02d}-weight'.format(i), '--{:02d}'.format(i),
                               action='store', type=float, default=None)

    argparser.add_argument('--adversarial-batch-size', '-B', action='store', type=int, default=32)
    argparser.add_argument('--adversarial-pooling', '-P', default='max', choices=['sum', 'max', 'mean', 'logsumexp'])

    argparser.add_argument('--report', '-r', default=10000, type=int,
                           help='Number of batches between performance reports')
    argparser.add_argument('--report-loss', default=100, type=int,
                           help='Number of batches between loss reports')

    argparser.add_argument('--eval', '-E', nargs='+', type=str, help='Evaluate on these additional sets')

    # Parameters for adversarial training
    argparser.add_argument('--lm', action='store', type=str, default='models/lm/',
                           help='Language Model')

    # XXX: default to None (disable) - 0.01
    argparser.add_argument('--adversarial-epsilon', '--aeps',
                           action='store', type=float, default=None)
    argparser.add_argument('--adversarial-nb-corruptions', '--anc',
                           action='store', type=int, default=32)
    argparser.add_argument('--adversarial-nb-examples-per-batch', '--anepb',
                           action='store', type=int, default=4)
    # XXX: default to -1 (disable) - 4
    argparser.add_argument('--adversarial-top-k', '--atopk',
                           action='store', type=int, default=-1)

    argparser.add_argument('--adversarial-flip', '--af', action='store_true', default=False)
    argparser.add_argument('--adversarial-combine', '--ac', action='store_true', default=False)
    argparser.add_argument('--adversarial-remove', '--ar', action='store_true', default=False)

    args = argparser.parse_args(argv)

    lm_path = args.lm
    a_epsilon = args.adversarial_epsilon
    a_nb_corr = args.adversarial_nb_corruptions
    a_nb_examples_per_batch = args.adversarial_nb_examples_per_batch
    a_top_k = args.adversarial_top_k
    a_is_flip = args.adversarial_flip
    a_is_combine = args.adversarial_combine
    a_is_remove = args.adversarial_remove

    # Command line arguments
    train_path, valid_path, test_path = args.train, args.valid, args.test

    model_name = args.model
    optimizer_name = args.optimizer

    embedding_size = args.embedding_size
    representation_size = args.representation_size

    batch_size = args.batch_size

    nb_epochs = args.nb_epochs
    nb_discriminator_epochs = args.nb_discriminator_epochs

    dropout_keep_prob = args.dropout_keep_prob
    learning_rate = args.learning_rate
    clip_value = args.clip
    seed = args.seed
    std_dev = args.std_dev

    has_bos = args.has_bos
    has_eos = args.has_eos
    has_unk = args.has_unk
    is_lower = args.lower

    initialize_embeddings = args.initialize_embeddings

    is_fixed_embeddings = args.fixed_embeddings
    is_normalize_embeddings = args.normalize_embeddings
    is_only_use_pretrained_embeddings = args.only_use_pretrained_embeddings
    is_semi_sort = args.semi_sort

    logger.info('Hostname: {}'.format(socket.gethostname()))

    logger.info('has_bos: {}, has_eos: {}, has_unk: {}'.format(has_bos, has_eos, has_unk))
    logger.info('is_lower: {}, is_fixed_embeddings: {}, is_normalize_embeddings: {}'
                .format(is_lower, is_fixed_embeddings, is_normalize_embeddings))
    logger.info('is_only_use_pretrained_embeddings: {}, is_semi_sort: {}'
                .format(is_only_use_pretrained_embeddings, is_semi_sort))

    save_path = args.save
    hard_save_path = args.hard_save
    restore_path = args.restore

    glove_path = args.glove

    # Experimental RTE regularizers
    rule00_weight = args.rule00_weight
    rule01_weight = args.rule01_weight
    rule02_weight = args.rule02_weight
    rule03_weight = args.rule03_weight
    rule04_weight = args.rule04_weight
    rule05_weight = args.rule05_weight
    rule06_weight = args.rule06_weight
    rule07_weight = args.rule07_weight
    rule08_weight = args.rule08_weight
    rule09_weight = args.rule09_weight
    rule10_weight = args.rule10_weight
    rule11_weight = args.rule11_weight
    rule12_weight = args.rule12_weight

    adversarial_pooling_name = args.adversarial_pooling

    name_to_adversarial_pooling = {
        'sum': tf.reduce_sum,
        'max': tf.reduce_max,
        'mean': tf.reduce_mean,
        'logsumexp': tf.reduce_logsumexp
    }

    report_interval = args.report
    report_loss_interval = args.report_loss

    eval_paths = args.eval

    np.random.seed(seed)
    rs = np.random.RandomState(seed)
    tf.set_random_seed(seed)

    logger.debug('Reading corpus ..')
    train_is, dev_is, test_is = util.SNLI.generate(train_path=train_path, valid_path=valid_path, test_path=test_path, is_lower=is_lower)

    logger.info('Train size: {}\tDev size: {}\tTest size: {}'.format(len(train_is), len(dev_is), len(test_is)))
    all_is = train_is + dev_is + test_is

    # Enumeration of tokens start at index=3:
    # index=0 PADDING, index=1 START_OF_SENTENCE, index=2 END_OF_SENTENCE, index=3 UNKNOWN_WORD
    bos_idx, eos_idx, unk_idx = 1, 2, 3
    start_idx = 1 + (1 if has_bos else 0) + (1 if has_eos else 0) + (1 if has_unk else 0)

    if not restore_path:
        # Create a sequence of tokens containing all sentences in the dataset
        token_seq = []
        for instance in all_is:
            token_seq += instance['sentence1_parse_tokens'] + instance['sentence2_parse_tokens']

        token_set = set(token_seq)
        allowed_words = None
        if is_only_use_pretrained_embeddings:
            assert glove_path is not None
            logger.info('Loading GloVe words from {}'.format(glove_path))
            assert os.path.isfile(glove_path)
            allowed_words = load_glove_words(path=glove_path, words=token_set)
            logger.info('Number of allowed words: {}'.format(len(allowed_words)))

        # Count the number of occurrences of each token
        token_counts = dict()
        for token in token_seq:
            if (allowed_words is None) or (token in allowed_words):
                if token not in token_counts:
                    token_counts[token] = 0
                token_counts[token] += 1

        # Sort the tokens according to their frequency and lexicographic ordering
        sorted_vocabulary = sorted(token_counts.keys(), key=lambda t: (- token_counts[t], t))

        index_to_token = {index: token for index, token in enumerate(sorted_vocabulary, start=start_idx)}
    else:
        vocab_path = '{}_index_to_token.p'.format(restore_path)
        logger.info('Restoring vocabulary from {} ..'.format(vocab_path))

        with open(vocab_path, 'rb') as f:
            index_to_token = pickle.load(f)

    token_to_index = {token: index for index, token in index_to_token.items()}

    entailment_idx, neutral_idx, contradiction_idx = 0, 1, 2
    label_to_index = {
        'entailment': entailment_idx,
        'neutral': neutral_idx,
        'contradiction': contradiction_idx,
    }

    max_len = None
    optimizer_name_to_class = {
        'adagrad': tf.train.AdagradOptimizer,
        'adam': tf.train.AdamOptimizer
    }

    optimizer_class = optimizer_name_to_class[optimizer_name]
    assert optimizer_class

    optimizer = optimizer_class(learning_rate=learning_rate)

    args = dict(has_bos=has_bos, has_eos=has_eos, has_unk=has_unk,
                bos_idx=bos_idx, eos_idx=eos_idx, unk_idx=unk_idx,
                max_len=max_len)

    train_dataset = util.instances_to_dataset(train_is, token_to_index, label_to_index, **args)
    dev_dataset = util.instances_to_dataset(dev_is, token_to_index, label_to_index, **args)
    test_dataset = util.instances_to_dataset(test_is, token_to_index, label_to_index, **args)

    sentence1 = train_dataset['sentence1']
    sentence1_length = train_dataset['sentence1_length']

    sentence2 = train_dataset['sentence2']
    sentence2_length = train_dataset['sentence2_length']

    label = train_dataset['label']

    sentence1_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence1')
    sentence2_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence2')

    sentence1_len_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence1_length')
    sentence2_len_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence2_length')

    clipped_sentence1 = tfutil.clip_sentence(sentence1_ph, sentence1_len_ph)
    clipped_sentence2 = tfutil.clip_sentence(sentence2_ph, sentence2_len_ph)

    label_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='label')

    token_set = set(token_to_index.keys())
    vocab_size = max(token_to_index.values()) + 1

    token_to_embedding = dict()
    if not restore_path:
        if glove_path:
            logger.info('Loading GloVe word embeddings from {}'.format(glove_path))
            assert os.path.isfile(glove_path)
            token_to_embedding = load_glove(glove_path, token_set)

    discriminator_scope_name = 'discriminator'
    with tf.variable_scope(discriminator_scope_name):
        if initialize_embeddings == 'normal':
            logger.info('Initializing the embeddings with 𝓝(0, 1)')
            embedding_initializer = tf.random_normal_initializer(0.0, 1.0)
        elif initialize_embeddings == 'uniform':
            logger.info('Initializing the embeddings with 𝒰(-1, 1)')
            embedding_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
        else:
            logger.info('Initializing the embeddings with Xavier initialization')
            embedding_initializer = tf.contrib.layers.xavier_initializer()

        embedding_layer = tf.get_variable('embeddings', shape=[vocab_size, embedding_size],
                                          initializer=embedding_initializer, trainable=not is_fixed_embeddings)

        sentence1_embedding = tf.nn.embedding_lookup(embedding_layer, clipped_sentence1)
        sentence2_embedding = tf.nn.embedding_lookup(embedding_layer, clipped_sentence2)

        dropout_keep_prob_ph = tf.placeholder(tf.float32, name='dropout_keep_prob')

        model_kwargs = dict(
            sequence1=sentence1_embedding, sequence1_length=sentence1_len_ph,
            sequence2=sentence2_embedding, sequence2_length=sentence2_len_ph,
            representation_size=representation_size, dropout_keep_prob=dropout_keep_prob_ph)

        if model_name in {'ff-dam', 'ff-damp', 'ff-dams'}:
            model_kwargs['init_std_dev'] = std_dev

        mode_name_to_class = {
            'cbilstm': ConditionalBiLSTM,
            'ff-dam': FeedForwardDAM,
            'ff-damp': FeedForwardDAMP,
            'ff-dams': FeedForwardDAMS,
            'esim1': ESIMv1
        }

        model_class = mode_name_to_class[model_name]

        assert model_class is not None
        model = model_class(**model_kwargs)

        logits = model()
        predictions = tf.argmax(logits, axis=1, name='predictions')

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label_ph)
        loss = tf.reduce_mean(losses)

        a_pooling_function = name_to_adversarial_pooling[adversarial_pooling_name]

        a_model_kwargs = copy.copy(model_kwargs)

        a_sentence1_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='a_sentence1')
        a_sentence2_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='a_sentence2')

        a_sentence1_len_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='a_sentence1_length')
        a_sentence2_len_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='a_sentence2_length')

        a_clipped_sentence1 = tfutil.clip_sentence(a_sentence1_ph, a_sentence1_len_ph)
        a_clipped_sentence2 = tfutil.clip_sentence(a_sentence2_ph, a_sentence2_len_ph)

        a_sentence1_embedding = tf.nn.embedding_lookup(embedding_layer, a_clipped_sentence1)
        a_sentence2_embedding = tf.nn.embedding_lookup(embedding_layer, a_clipped_sentence2)

        a_model_kwargs.update({
            'sequence1': a_sentence1_embedding, 'sequence1_length': a_sentence1_len_ph,
            'sequence2': a_sentence2_embedding, 'sequence2_length': a_sentence2_len_ph
        })

        a_kwargs = dict(model_class=model_class, model_kwargs=a_model_kwargs,
                        entailment_idx=entailment_idx, contradiction_idx=contradiction_idx, neutral_idx=neutral_idx,
                        pooling_function=a_pooling_function, debug=True)

        a_losses = None
        a_function_weight_bi_tuple_lst = []

        if rule00_weight:
            a_loss, a_losses = R.contradiction_symmetry_l2(**a_kwargs)
            a_function_weight_bi_tuple_lst += [(R.contradiction_symmetry_l2, rule00_weight, False)]
            loss += rule00_weight * a_loss
        if rule01_weight:
            a_loss, a_losses = R.contradiction_symmetry_l1(**a_kwargs)
            a_function_weight_bi_tuple_lst += [(R.contradiction_symmetry_l1, rule01_weight, False)]
            loss += rule01_weight * a_loss
        if rule02_weight:
            a_loss, a_losses = R.contradiction_kullback_leibler(**a_kwargs)
            a_function_weight_bi_tuple_lst += [(R.contradiction_kullback_leibler, rule02_weight, False)]
            loss += rule02_weight * a_loss
        if rule03_weight:
            a_loss, a_losses = R.contradiction_jensen_shannon(**a_kwargs)
            a_function_weight_bi_tuple_lst += [(R.contradiction_jensen_shannon, rule03_weight, False)]
            loss += rule03_weight * a_loss

        if rule04_weight:
            a_loss, a_losses = R.contradiction_acl(**a_kwargs)
            a_function_weight_bi_tuple_lst += [(R.contradiction_acl, rule04_weight, False)]
            loss += rule04_weight * a_loss
        if rule05_weight:
            a_loss, a_losses = R.entailment_acl(**a_kwargs)
            a_function_weight_bi_tuple_lst += [(R.entailment_acl, rule05_weight, False)]
            loss += rule05_weight * a_loss
        if rule06_weight:
            a_loss, a_losses = R.neutral_acl(**a_kwargs)
            a_function_weight_bi_tuple_lst += [(R.neutral_acl, rule06_weight, False)]
            loss += rule06_weight * a_loss

        if rule07_weight:
            a_loss, a_losses = R.contradiction_acl(is_bi=True, **a_kwargs)
            a_function_weight_bi_tuple_lst += [(R.contradiction_acl, rule07_weight, True)]
            loss += rule07_weight * a_loss
        if rule08_weight:
            a_loss, a_losses = R.entailment_acl(is_bi=True, **a_kwargs)
            a_function_weight_bi_tuple_lst += [(R.entailment_acl, rule08_weight, True)]
            loss += rule08_weight * a_loss
        if rule09_weight:
            a_loss, a_losses = R.neutral_acl(is_bi=True, **a_kwargs)
            a_function_weight_bi_tuple_lst += [(R.neutral_acl, rule09_weight, True)]
            loss += rule09_weight * a_loss

        if rule10_weight:
            a_loss, a_losses = R.entailment_reflexive_acl(**a_kwargs)
            a_function_weight_bi_tuple_lst += [(R.entailment_reflexive_acl, rule10_weight, False)]
            loss += rule10_weight * a_loss
        if rule11_weight:
            a_loss, a_losses = R.entailment_neutral_acl(**a_kwargs)
            a_function_weight_bi_tuple_lst += [(R.entailment_neutral_acl, rule11_weight, False)]
            loss += rule11_weight * a_loss
        if rule12_weight:
            a_loss, a_losses = R.entailment_neutral_acl(is_bi=True, **a_kwargs)
            a_function_weight_bi_tuple_lst += [(R.entailment_neutral_acl, rule12_weight, True)]
            loss += rule12_weight * a_loss

    discriminator_vars = tfutil.get_variables_in_scope(discriminator_scope_name)
    discriminator_init_op = tf.variables_initializer(discriminator_vars)

    trainable_discriminator_vars = list(discriminator_vars)
    if is_fixed_embeddings:
        trainable_discriminator_vars.remove(embedding_layer)

    discriminator_optimizer_scope_name = 'discriminator_optimizer'
    with tf.variable_scope(discriminator_optimizer_scope_name):
        if clip_value:
            gradients, v = zip(*optimizer.compute_gradients(loss, var_list=trainable_discriminator_vars))
            gradients, _ = tf.clip_by_global_norm(gradients, clip_value)
            training_step = optimizer.apply_gradients(zip(gradients, v))
        else:
            training_step = optimizer.minimize(loss, var_list=trainable_discriminator_vars)

    discriminator_optimizer_vars = tfutil.get_variables_in_scope(discriminator_optimizer_scope_name)
    discriminator_optimizer_init_op = tf.variables_initializer(discriminator_optimizer_vars)

    token_idx_ph = tf.placeholder(dtype=tf.int32, name='word_idx')
    token_embedding_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='word_embedding')

    assign_token_embedding = embedding_layer[token_idx_ph, :].assign(token_embedding_ph)

    init_projection_steps = []
    learning_projection_steps = []

    if is_normalize_embeddings:
        embeddings_projection = constraints.unit_sphere(embedding_layer, norm=1.0)
        init_projection_steps += [embeddings_projection]

        if not is_fixed_embeddings:
            learning_projection_steps += [embeddings_projection]

    predictions_int = tf.cast(predictions, tf.int32)
    labels_int = tf.cast(label_ph, tf.int32)

    a_feed_dict = dict()

    d_sentence1, d_sentence2 = train_dataset['sentence1'], train_dataset['sentence2']
    d_sentence1_len, d_sentence2_len = train_dataset['sentence1_length'], train_dataset['sentence2_length']
    d_label = train_dataset['label']

    nb_train_instances = d_label.shape[0]

    max_sentence_len = max(d_sentence1.shape[1], d_sentence2.shape[1])
    d_sentence = np.zeros(shape=(nb_train_instances * 2, max_sentence_len), dtype=np.int)
    d_sentence[0:d_sentence1.shape[0], 0:d_sentence1.shape[1]] = d_sentence1
    d_sentence[d_sentence1.shape[0]:, 0:d_sentence2.shape[1]] = d_sentence2

    d_sentence_len = np.concatenate((d_sentence1_len, d_sentence2_len), axis=0)

    saver = tf.train.Saver(discriminator_vars + discriminator_optimizer_vars, max_to_keep=1)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    a_batch_size = 1 + (a_nb_corr * a_is_flip) + (a_nb_corr * a_is_remove) + (a_nb_corr * a_is_combine)

    G = Generator(token_to_index=token_to_index,
                  nb_corruptions=a_nb_corr)

    IS = None
    if a_top_k is not None:
        with tf.variable_scope(discriminator_scope_name):
            IS = IScorer(embedding_layer=embedding_layer,
                         token_to_index=token_to_index,
                         model_class=model_class,
                         model_kwargs=model_kwargs,
                         i_pooling_function=tf.reduce_sum,
                         a_function_weight_bi_tuple_lst=a_function_weight_bi_tuple_lst)

    LMS = None
    if a_epsilon is not None:
        LMS = LMScorer(embedding_layer=embedding_layer,
                       token_to_index=token_to_index,
                       batch_size=a_batch_size)

        lm_vars = LMS.get_vars()
        lm_saver = tf.train.Saver(lm_vars, max_to_keep=1)

    A_rs = np.random.RandomState(0)

    with tf.Session(config=session_config) as session:
        logger.info('Total Parameters: {}'.format(tfutil.count_trainable_parameters()))
        logger.info('Total Discriminator Parameters: {}'.format(
            tfutil.count_trainable_parameters(var_list=discriminator_vars)))
        logger.info('Total Trainable Discriminator Parameters: {}'.format(
            tfutil.count_trainable_parameters(var_list=trainable_discriminator_vars)))

        if LMS is not None:
            lm_ckpt = tf.train.get_checkpoint_state(lm_path)
            lm_saver.restore(session, lm_ckpt.model_checkpoint_path)

        if restore_path:
            saver.restore(session, restore_path)
        else:
            session.run([discriminator_init_op, discriminator_optimizer_init_op])

            # Initialising pre-trained embeddings
            logger.info('Initialising the embeddings pre-trained vectors ..')
            for token in token_to_embedding:
                token_idx, token_embedding = token_to_index[token], token_to_embedding[token]
                assert embedding_size == len(token_embedding)
                session.run(assign_token_embedding,
                            feed_dict={
                                token_idx_ph: token_idx,
                                token_embedding_ph: token_embedding
                            })
            logger.info('Done!')

            for adversary_projection_step in init_projection_steps:
                session.run([adversary_projection_step])

        nb_instances = sentence1.shape[0]
        batches = make_batches(size=nb_instances, batch_size=batch_size)

        best_dev_acc, best_test_acc = None, None
        discriminator_batch_counter = 0

        for epoch in range(1, nb_epochs + 1):
            for d_epoch in range(1, nb_discriminator_epochs + 1):
                order = rs.permutation(nb_instances)

                sentences1, sentences2 = sentence1[order], sentence2[order]
                sizes1, sizes2 = sentence1_length[order], sentence2_length[order]
                labels = label[order]

                if is_semi_sort:
                    order = util.semi_sort(sizes1, sizes2)
                    sentences1, sentences2 = sentence1[order], sentence2[order]
                    sizes1, sizes2 = sentence1_length[order], sentence2_length[order]
                    labels = label[order]

                loss_values, epoch_loss_values = [], []
                for batch_idx, (batch_start, batch_end) in enumerate(batches):
                    discriminator_batch_counter += 1

                    batch_sentences1 = sentences1[batch_start:batch_end]
                    batch_sentences2 = sentences2[batch_start:batch_end]

                    batch_sizes1 = sizes1[batch_start:batch_end]
                    batch_sizes2 = sizes2[batch_start:batch_end]

                    batch_labels = labels[batch_start:batch_end]

                    batch_max_size1 = np.max(batch_sizes1)
                    batch_max_size2 = np.max(batch_sizes2)

                    batch_sentences1 = batch_sentences1[:, :batch_max_size1]
                    batch_sentences2 = batch_sentences2[:, :batch_max_size2]

                    # TODO: This is where we generate, score, and select the adversarial examples

                    # Remove the BOS
                    o_batch_size = batch_sentences1.shape[0]
                    o_sentences1, o_sentences2 = [], []

                    for i in range(o_batch_size):
                        _start_idx = 1 if has_bos else 0
                        o_sentences1 += [[idx for idx in batch_sentences1[i, _start_idx:].tolist() if idx != 0]]
                        o_sentences2 += [[idx for idx in batch_sentences2[i, _start_idx:].tolist() if idx != 0]]

                    # Parameters for adversarial training:
                    # a_epsilon, a_nb_corruptions, a_nb_examples_per_batch, a_is_flip, a_is_combine, a_is_remove
                    selected_sentence1, selected_sentence2 = [], []

                    c_idxs = A_rs.choice(o_batch_size, a_nb_examples_per_batch, replace=False)
                    for c_idx in c_idxs:
                        o_sentence1 = o_sentences1[c_idx]
                        o_sentence2 = o_sentences2[c_idx]

                        # Generating Corruptions
                        c_sentence1_lst, c_sentence2_lst = [o_sentence1], [o_sentence2]
                        if a_is_flip:
                            corr1, corr2 = G.flip(sentence1=o_sentence1, sentence2=o_sentence2)
                            c_sentence1_lst += corr1
                            c_sentence2_lst += corr2
                        if a_is_remove:
                            corr1, corr2 = G.remove(sentence1=o_sentence1, sentence2=o_sentence2)
                            c_sentence1_lst += corr1
                            c_sentence2_lst += corr2
                        if a_is_combine:
                            corr1, corr2 = G.combine(sentence1=o_sentence1, sentence2=o_sentence2)
                            c_sentence1_lst += corr1
                            c_sentence2_lst += corr2

                        if a_epsilon is not None and LMS is not None:
                            # Scoring them against a Language Model
                            log_perp1 = LMS.score(session, c_sentence1_lst)
                            log_perp2 = LMS.score(session, c_sentence2_lst)

                            low_lperp_idxs = np.where(
                                (log_perp1 + log_perp2) < (log_perp1[0] + log_perp2[0] + a_epsilon)
                            )[0].tolist()
                        else:
                            low_lperp_idxs = range(len(c_sentence1_lst))

                        selected_sentence1 += [c_sentence1_lst[i] for i in low_lperp_idxs]
                        selected_sentence2 += [c_sentence2_lst[i] for i in low_lperp_idxs]

                    # Now in selected_sentence1 and selected_sentence2 we have the most offending examples
                    if a_top_k >= 0 and IS is not None:
                        iscore_values = IS.iscore(session, selected_sentence1, selected_sentence2)
                        top_k_idxs = np.argsort(iscore_values)[::-1][:a_top_k]
                        selected_sentence1 = [selected_sentence1[i] for i in top_k_idxs]
                        selected_sentence2 = [selected_sentence2[i] for i in top_k_idxs]

                    # Convert selected_sentence1 and selected_sentence2 into a batch
                    batch_a_sentences1 = util.pad_sequences(selected_sentence1)
                    batch_a_sentences2 = util.pad_sequences(selected_sentence2)

                    batch_a_sizes1 = np.array([len(s) for s in selected_sentence1])
                    batch_a_sizes2 = np.array([len(s) for s in selected_sentence2])

                    if not (batch_sentences1.shape[0] == batch_a_sentences2.shape[0] == a_batch_size):
                        logger.error("{} {} {}".format(
                            str(batch_sentences1.shape),
                            str(batch_a_sentences2.shape),
                            str(a_batch_size)))
                    assert batch_sentences1.shape[0] == batch_a_sentences2.shape[0] == a_batch_size

                    batch_feed_dict = {
                        sentence1_ph: batch_sentences1, sentence1_len_ph: batch_sizes1,
                        sentence2_ph: batch_sentences2, sentence2_len_ph: batch_sizes2,
                        label_ph: batch_labels, dropout_keep_prob_ph: dropout_keep_prob,

                        a_sentence1_ph: batch_a_sentences1, a_sentence1_len_ph: batch_a_sizes1,
                        a_sentence2_ph: batch_a_sentences2, a_sentence2_len_ph: batch_a_sizes2
                    }

                    # Adding the adversaries
                    batch_feed_dict.update(a_feed_dict)

                    _, loss_value = session.run([training_step, loss], feed_dict=batch_feed_dict)

                    logger.debug('Epoch {0}/{1}/{2}\tLoss: {3}'.format(epoch, d_epoch, batch_idx, loss_value))

                    cur_batch_size = batch_sentences1.shape[0]
                    loss_values += [loss_value / cur_batch_size]
                    epoch_loss_values += [loss_value / cur_batch_size]

                    for adversary_projection_step in learning_projection_steps:
                        session.run([adversary_projection_step])

                    if discriminator_batch_counter % report_loss_interval == 0:
                        logger.info('Epoch {0}/{1}/{2}\tLoss Stats: {3}'.format(epoch, d_epoch, batch_idx, stats(loss_values)))
                        loss_values = []

                    if discriminator_batch_counter % report_interval == 0:
                        accuracy_args = [sentence1_ph, sentence1_len_ph, sentence2_ph, sentence2_len_ph,
                                         label_ph, dropout_keep_prob_ph, predictions_int, labels_int,
                                         contradiction_idx, entailment_idx, neutral_idx, batch_size]
                        dev_acc, _, _, _ = accuracy(session, dev_dataset, 'Dev', *accuracy_args)
                        test_acc, _, _, _ = accuracy(session, test_dataset, 'Test', *accuracy_args)

                        logger.info('Epoch {0}/{1}/{2}\tDev Acc: {3:.2f}\tTest Acc: {4:.2f}'
                                    .format(epoch, d_epoch, batch_idx, dev_acc * 100, test_acc * 100))

                        if best_dev_acc is None or dev_acc > best_dev_acc:
                            best_dev_acc, best_test_acc = dev_acc, test_acc

                            if save_path:
                                with open('{}_index_to_token.p'.format(save_path), 'wb') as f:
                                    pickle.dump(index_to_token, f)

                                saved_path = saver.save(session, save_path)
                                logger.info('Model saved in file: {}'.format(saved_path))

                        logger.info('Epoch {0}/{1}/{2}\tBest Dev Accuracy: {3:.2f}\tBest Test Accuracy: {4:.2f}'
                                    .format(epoch, d_epoch, batch_idx, best_dev_acc * 100, best_test_acc * 100))

                        if eval_paths is not None:
                            for eval_path in eval_paths:
                                eval_path_acc = eutil.evaluate(session, eval_path, label_to_index, token_to_index, predictions, batch_size,
                                                               sentence1_ph, sentence2_ph, sentence1_len_ph, sentence2_len_ph, dropout_keep_prob_ph,
                                                               has_bos=has_bos, has_eos=has_eos, has_unk=has_unk, is_lower=is_lower,
                                                               bos_idx=bos_idx, eos_idx=eos_idx, unk_idx=unk_idx)
                                logger.info('Epoch {0}/{1}/{2}\tAccuracy on {3} is {4}'.format(epoch, d_epoch, batch_idx,
                                                                                               eval_path, eval_path_acc))

                        if a_losses is not None:
                            t_feed_dict = a_feed_dict
                            if len(t_feed_dict) == 0:
                                t_feed_dict = {
                                    sentence1_ph: sentences1[:1024], sentence1_len_ph: sizes1[:1024],
                                    sentence2_ph: sentences2[:1024], sentence2_len_ph: sizes2[:1024],
                                    dropout_keep_prob_ph: 1.0
                                }
                            a_losses_value = session.run(a_losses, feed_dict=t_feed_dict)

                            a_input_idxs = np.argsort(- a_losses_value)
                            for i in a_input_idxs[:10]:
                                t_sentence1 = t_feed_dict[sentence1_ph][i]
                                t_sentence2 = t_feed_dict[sentence2_ph][i]

                                logger.info('[ {} / {} ] Sentence1: {}'.format(i, a_losses_value[i], ' '.join([index_to_token[x] for x in t_sentence1 if x not in [0, 1, 2]])))
                                logger.info('[ {} / {} ] Sentence2: {}'.format(i, a_losses_value[i], ' '.join([index_to_token[x] for x in t_sentence2 if x not in [0, 1, 2]])))

                logger.info('Epoch {0}/{1}\tEpoch Loss Stats: {2}'.format(epoch, d_epoch, stats(epoch_loss_values)))

                if hard_save_path:
                    with open('{}_index_to_token.p'.format(hard_save_path), 'wb') as f:
                        pickle.dump(index_to_token, f)

                    hard_saved_path = saver.save(session, hard_save_path)
                    logger.info('Model saved in file: {}'.format(hard_saved_path))

    logger.info('Training finished.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
