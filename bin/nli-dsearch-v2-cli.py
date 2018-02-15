#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import os
import sys
import json

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

    def fmt(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Regularising RTE via Adversarial Sets Regularisation', formatter_class=fmt)

    argparser.add_argument('--data', '-d', action='store', type=str, default='data/snli/snli_1.0_train.jsonl.gz')

    argparser.add_argument('--model', '-m', action='store', type=str, default='ff-dam',
                           choices=['cbilstm', 'ff-dam', 'ff-damp', 'ff-dams', 'esim1'])

    argparser.add_argument('--embedding-size', action='store', type=int, default=300)
    argparser.add_argument('--representation-size', action='store', type=int, default=200)

    argparser.add_argument('--batch-size', action='store', type=int, default=32)

    argparser.add_argument('--seed', action='store', type=int, default=0)

    argparser.add_argument('--has-bos', action='store_true', default=False, help='Has <Beginning Of Sentence> token')
    argparser.add_argument('--has-eos', action='store_true', default=False, help='Has <End Of Sentence> token')
    argparser.add_argument('--has-unk', action='store_true', default=False, help='Has <Unknown Word> token')

    argparser.add_argument('--semi-sort', '-S', action='store_true')

    argparser.add_argument('--restore', action='store', type=str, default=None)

    for i in range(0, 13):
        argparser.add_argument('--rule{:02d}-weight'.format(i), '--{:02d}'.format(i),
                               action='store', type=float, default=None)

    argparser.add_argument('--adversarial-batch-size', '-B', action='store', type=int, default=32)
    argparser.add_argument('--adversarial-pooling', '-P', default='max', choices=['sum', 'max', 'mean', 'logsumexp'])

    argparser.add_argument('--report', '-r', default=10000, type=int,
                           help='Number of batches between performance reports')
    argparser.add_argument('--report-loss', default=100, type=int,
                           help='Number of batches between loss reports')

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

    argparser.add_argument('--json', action='store', type=str, default=None)

    args = argparser.parse_args(argv)

    lm_path = args.lm
    a_epsilon = args.adversarial_epsilon
    a_nb_corr = args.adversarial_nb_corruptions
    a_nb_examples_per_batch = args.adversarial_nb_examples_per_batch
    a_top_k = args.adversarial_top_k
    a_is_flip = args.adversarial_flip
    a_is_combine = args.adversarial_combine
    a_is_remove = args.adversarial_remove

    json_path = args.json

    # Command line arguments
    data_path = args.data

    model_name = args.model

    embedding_size = args.embedding_size
    representation_size = args.representation_size

    batch_size = args.batch_size

    seed = args.seed

    has_bos = args.has_bos
    has_eos = args.has_eos
    has_unk = args.has_unk

    restore_path = args.restore

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

    np.random.seed(seed)
    rs = np.random.RandomState(seed)
    tf.set_random_seed(seed)

    logger.debug('Reading corpus ..')
    data_is, _, _ = util.SNLI.generate(train_path=data_path)

    logger.info('Data size: {}'.format(len(data_is)))

    # Enumeration of tokens start at index=3:
    # index=0 PADDING, index=1 START_OF_SENTENCE, index=2 END_OF_SENTENCE, index=3 UNKNOWN_WORD
    bos_idx, eos_idx, unk_idx = 1, 2, 3
    start_idx = 1 + (1 if has_bos else 0) + (1 if has_eos else 0) + (1 if has_unk else 0)

    assert restore_path is not None
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
    index_to_label = {k: v for v, k in label_to_index.items()}

    max_len = None
    args = dict(has_bos=has_bos, has_eos=has_eos, has_unk=has_unk,
                bos_idx=bos_idx, eos_idx=eos_idx, unk_idx=unk_idx,
                max_len=max_len)

    dataset = util.instances_to_dataset(data_is, token_to_index, label_to_index, **args)

    sentence1 = dataset['sentence1']
    sentence1_length = dataset['sentence1_length']

    sentence2 = dataset['sentence2']
    sentence2_length = dataset['sentence2_length']

    label = dataset['label']

    sentence1_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence1')
    sentence2_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence2')

    sentence1_len_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence1_length')
    sentence2_len_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence2_length')

    clipped_sentence1 = tfutil.clip_sentence(sentence1_ph, sentence1_len_ph)
    clipped_sentence2 = tfutil.clip_sentence(sentence2_ph, sentence2_len_ph)

    nb_instances = sentence1.shape[0]
    token_set = set(token_to_index.keys())
    vocab_size = max(token_to_index.values()) + 1

    discriminator_scope_name = 'discriminator'
    with tf.variable_scope(discriminator_scope_name):

        embedding_layer = tf.get_variable('embeddings',
                                          shape=[vocab_size, embedding_size],
                                          initializer=None)

        sentence1_embedding = tf.nn.embedding_lookup(embedding_layer, clipped_sentence1)
        sentence2_embedding = tf.nn.embedding_lookup(embedding_layer, clipped_sentence2)

        model_kwargs = dict(
            sequence1=sentence1_embedding, sequence1_length=sentence1_len_ph,
            sequence2=sentence2_embedding, sequence2_length=sentence2_len_ph,
            representation_size=representation_size, dropout_keep_prob=1.0)

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
        probabilities = tf.nn.softmax(logits)

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

        a_function_weight_bi_tuple_lst = []

        loss = tf.constant(0.0)

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
    trainable_discriminator_vars.remove(embedding_layer)

    saver = tf.train.Saver(discriminator_vars, max_to_keep=1)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

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

    a_batch_size = (a_nb_corr * a_is_flip) + (a_nb_corr * a_is_remove) + (a_nb_corr * a_is_combine)

    LMS = None
    if a_epsilon is not None:
        LMS = LMScorer(embedding_layer=embedding_layer,
                       token_to_index=token_to_index,
                       batch_size=a_batch_size)

        lm_vars = LMS.get_vars()
        lm_saver = tf.train.Saver(lm_vars, max_to_keep=1)

    A_rs = np.random.RandomState(0)

    sp2op = {}
    op2lbl = {}

    with tf.Session(config=session_config) as session:

        if LMS is not None:
            lm_ckpt = tf.train.get_checkpoint_state(lm_path)
            lm_saver.restore(session, lm_ckpt.model_checkpoint_path)

        saver.restore(session, restore_path)

        batches = make_batches(size=nb_instances, batch_size=batch_size)

        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            order = np.arange(nb_instances)

            sentences1, sentences2 = sentence1[order], sentence2[order]
            sizes1, sizes2 = sentence1_length[order], sentence2_length[order]
            labels = label[order]

            batch_sentences1 = sentences1[batch_start:batch_end]
            batch_sentences2 = sentences2[batch_start:batch_end]

            batch_sizes1 = sizes1[batch_start:batch_end]
            batch_sizes2 = sizes2[batch_start:batch_end]

            batch_labels = labels[batch_start:batch_end]

            batch_max_size1 = np.max(batch_sizes1)
            batch_max_size2 = np.max(batch_sizes2)

            batch_sentences1 = batch_sentences1[:, :batch_max_size1]
            batch_sentences2 = batch_sentences2[:, :batch_max_size2]

            # This is where we generate, score, and select the adversarial examples
            cur_batch_size = batch_sentences1.shape[0]

            # Remove the BOS token from sentences
            o_batch_size = batch_sentences1.shape[0]
            o_sentences1, o_sentences2 = [], []

            for i in range(o_batch_size):
                _start_idx = 1 if has_bos else 0
                o_sentences1 += [[idx for idx in batch_sentences1[i, _start_idx:].tolist() if idx != 0]]
                o_sentences2 += [[idx for idx in batch_sentences2[i, _start_idx:].tolist() if idx != 0]]

            # Parameters for adversarial training:
            # a_epsilon, a_nb_corruptions, a_nb_examples_per_batch, a_is_flip, a_is_combine, a_is_remove
            selected_sentence1, selected_sentence2 = [], []

            # First, add all training sentences
            selected_sentence1 += o_sentences1
            selected_sentence2 += o_sentences2

            for a, b, c in zip(selected_sentence1, selected_sentence2, batch_labels):
                sp2op[(tuple(a), tuple(b))] = (a, b)
                op2lbl[(tuple(a), tuple(b))] = c

            c_idxs = A_rs.choice(o_batch_size, a_nb_examples_per_batch, replace=False)
            for c_idx in c_idxs:
                o_sentence1 = o_sentences1[c_idx]
                o_sentence2 = o_sentences2[c_idx]

                sp2op[(tuple(o_sentence1), tuple(o_sentence2))] = (o_sentence1, o_sentence2)

                # Generating Corruptions
                c_sentence1_lst, c_sentence2_lst = [], []
                if a_is_flip:
                    corr1, corr2 = G.flip(sentence1=o_sentence1, sentence2=o_sentence2)
                    c_sentence1_lst += corr1
                    c_sentence2_lst += corr2

                    for _c1, _c2 in zip(corr1, corr2):
                        sp2op[(tuple(_c1), tuple(_c2))] = (o_sentence1, o_sentence2)

                if a_is_remove:
                    corr1, corr2 = G.remove(sentence1=o_sentence1, sentence2=o_sentence2)
                    c_sentence1_lst += corr1
                    c_sentence2_lst += corr2

                    for _c1, _c2 in zip(corr1, corr2):
                        sp2op[(tuple(_c1), tuple(_c2))] = (o_sentence1, o_sentence2)

                if a_is_combine:
                    corr1, corr2 = G.combine(sentence1=o_sentence1, sentence2=o_sentence2)
                    c_sentence1_lst += corr1
                    c_sentence2_lst += corr2

                    for _c1, _c2 in zip(corr1, corr2):
                        sp2op[(tuple(_c1), tuple(_c2))] = (o_sentence1, o_sentence2)

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

            selected_scores = None
            # Now in selected_sentence1 and selected_sentence2 we have the most offending examples
            if a_top_k >= 0 and IS is not None:
                iscore_values = IS.iscore(session, selected_sentence1, selected_sentence2)
                top_k_idxs = np.argsort(iscore_values)[::-1][:a_top_k]

                selected_sentence1 = [selected_sentence1[i] for i in top_k_idxs]
                selected_sentence2 = [selected_sentence2[i] for i in top_k_idxs]

                selected_scores = [iscore_values[i] for i in top_k_idxs]

            def decode(sentence_ids):
                return ' '.join([index_to_token[idx] for idx in sentence_ids])

            def infer(s1_ids, s2_ids):
                a = np.array([[bos_idx] + s1_ids])
                b = np.array([[bos_idx] + s2_ids])

                c = np.array([1 + len(s1_ids)])
                d = np.array([1 + len(s2_ids)])

                inf_feed = {
                    sentence1_ph: a, sentence2_ph: b,
                    sentence1_len_ph: c, sentence2_len_ph: d
                }
                pv = session.run(probabilities, feed_dict=inf_feed)
                return {
                    'ent': str(pv[0, entailment_idx]),
                    'neu': str(pv[0, neutral_idx]),
                    'con': str(pv[0, contradiction_idx])
                }

            logger.info("No. of generated pairs: {}".format(len(selected_sentence1)))

            for i, (s1, s2, score) in enumerate(zip(selected_sentence1, selected_sentence2, selected_scores)):
                o1, o2 = sp2op[(tuple(s1), tuple(s2))]
                lbl = op2lbl[(tuple(o1), tuple(o2))]

                print('[{}] Original 1: {}'.format(i, decode(o1)))
                print('[{}] Original 2: {}'.format(i, decode(o2)))
                print('[{}] Original Label: {}'.format(i, index_to_label[lbl]))

                print('[{}] Sentence 1: {}'.format(i, decode(s1)))
                print('[{}] Sentence 2: {}'.format(i, decode(s2)))

                print('[{}] Inconsistency Loss: {}'.format(i, score))

                print('[{}] (before) s1 -> s2: {}'.format(i, str(infer(o1, o2))))
                print('[{}] (before) s2 -> s1: {}'.format(i, str(infer(o2, o1))))

                print('[{}] (after) s1 -> s2: {}'.format(i, str(infer(s1, s2))))
                print('[{}] (after) s2 -> s1: {}'.format(i, str(infer(s2, s1))))

                jdata = {
                    'original_sentence1': decode(o1),
                    'original_sentence2': decode(o2),
                    'original_label': index_to_label[lbl],
                    'sentence1': decode(s1),
                    'sentence1': decode(s2),
                    'inconsistency_loss': str(score),
                    'probabilities_before_s1_s2': infer(o1, o2),
                    'probabilities_before_s2_s1': infer(o2, o1),
                    'probabilities_after_s1_s2': infer(s1, s2),
                    'probabilities_after_s2_s1': infer(s2, s1)
                }

                if json_path is not None:
                    with open(json_path, 'a') as f:
                        json.dump(jdata, f)
                        f.write('\n')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
