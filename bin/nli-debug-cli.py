#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import os
import sys

import pickle

import numpy as np
import tensorflow as tf

from inferbeddings.models.training.util import make_batches

from tqdm import tqdm

from inferbeddings.nli import util, tfutil
from inferbeddings.nli import ConditionalBiLSTM
from inferbeddings.nli import FeedForwardDAM
from inferbeddings.nli import FeedForwardDAMP
from inferbeddings.nli import FeedForwardDAMS
from inferbeddings.nli import ESIMv1


import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


# Running:
#  $ python3 ./bin/nli-debug-cli.py -m ff-dam --batch-size 32 --representation-size 200 --has-bos --has-unk
#    --restore models/snli/dam_1/dam_1

def main(argv):
    logger.info('Command line: {}'.format(' '.join(arg for arg in argv)))

    def fmt(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Regularising RTE via Adversarial Sets Regularisation', formatter_class=fmt)

    argparser.add_argument('--data', '-d', action='store', type=str, default='data/snli/snli_1.0_train.jsonl.gz')
    argparser.add_argument('--model', '-m', action='store', type=str, default='cbilstm',
                           choices=['cbilstm', 'ff-dam', 'ff-damp', 'ff-dams', 'esim1'])

    argparser.add_argument('--embedding-size', action='store', type=int, default=300)
    argparser.add_argument('--representation-size', action='store', type=int, default=200)

    argparser.add_argument('--batch-size', action='store', type=int, default=1024)

    argparser.add_argument('--seed', action='store', type=int, default=0)

    argparser.add_argument('--has-bos', action='store_true', default=False, help='Has <Beginning Of Sentence> token')
    argparser.add_argument('--has-eos', action='store_true', default=False, help='Has <End Of Sentence> token')
    argparser.add_argument('--has-unk', action='store_true', default=False, help='Has <Unknown Word> token')
    argparser.add_argument('--lower', '-l', action='store_true', default=False, help='Lowercase the corpus')

    argparser.add_argument('--restore', action='store', type=str, default=None)

    args = argparser.parse_args(argv)

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
    is_lower = args.lower

    restore_path = args.restore

    np.random.seed(seed)
    rs = np.random.RandomState(seed)
    tf.set_random_seed(seed)

    logger.debug('Reading corpus ..')
    data_is, _, _ = util.SNLI.generate(train_path=data_path, valid_path=None, test_path=None, is_lower=is_lower)

    logger.info('Data size: {}'.format(len(data_is)))

    # Enumeration of tokens start at index=3:
    # index=0 PADDING, index=1 START_OF_SENTENCE, index=2 END_OF_SENTENCE, index=3 UNKNOWN_WORD
    bos_idx, eos_idx, unk_idx = 1, 2, 3

    with open('{}_index_to_token.p'.format(restore_path), 'rb') as f:
        index_to_token = pickle.load(f)

    token_to_index = {token: index for index, token in index_to_token.items()}

    entailment_idx, neutral_idx, contradiction_idx = 0, 1, 2
    label_to_index = {
        'entailment': entailment_idx,
        'neutral': neutral_idx,
        'contradiction': contradiction_idx,
    }

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

    token_set = set(token_to_index.keys())
    vocab_size = max(token_to_index.values()) + 1

    discriminator_scope_name = 'discriminator'
    with tf.variable_scope(discriminator_scope_name):
        embedding_layer = tf.get_variable('embeddings',
                                          shape=[vocab_size, embedding_size],
                                          trainable=False)

        sentence1_embedding = tf.nn.embedding_lookup(embedding_layer, clipped_sentence1)
        sentence2_embedding = tf.nn.embedding_lookup(embedding_layer, clipped_sentence2)

        dropout_keep_prob_ph = tf.placeholder(tf.float32, name='dropout_keep_prob')

        model_kwargs = dict(
            sequence1=sentence1_embedding, sequence1_length=sentence1_len_ph,
            sequence2=sentence2_embedding, sequence2_length=sentence2_len_ph,
            representation_size=representation_size, dropout_keep_prob=dropout_keep_prob_ph)

        if model_name in {'ff-dam', 'ff-damp', 'ff-dams'}:
            model_kwargs['init_std_dev'] = 0.01

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

    discriminator_vars = tfutil.get_variables_in_scope(discriminator_scope_name)

    trainable_discriminator_vars = list(discriminator_vars)

    predictions_int = tf.cast(predictions, tf.int32)

    # Dataset, in the form of matrices and arrays
    d_sentence1, d_sentence2 = dataset['sentence1'], dataset['sentence2']
    d_sentence1_len, d_sentence2_len = dataset['sentence1_length'], dataset['sentence2_length']
    d_label = dataset['label']

    nb_train_instances = d_label.shape[0]
    max_sentence_len = max(d_sentence1.shape[1], d_sentence2.shape[1])

    # Single big matrix containing all sentences in the dataset
    d_sentence = np.zeros(shape=(nb_train_instances * 2, max_sentence_len), dtype=np.int)
    d_sentence[0:d_sentence1.shape[0], 0:d_sentence1.shape[1]] = d_sentence1
    d_sentence[d_sentence1.shape[0]:, 0:d_sentence2.shape[1]] = d_sentence2

    d_sentence_len = np.concatenate([d_sentence1_len, d_sentence2_len])
    assert d_sentence.shape[0] == d_sentence_len.shape[0]

    saver = tf.train.Saver(discriminator_vars, max_to_keep=1)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    with tf.Session(config=session_config) as session:
        logger.info('Total Parameters: {}'.format(
            tfutil.count_trainable_parameters()))

        logger.info('Total Discriminator Parameters: {}'.format(
            tfutil.count_trainable_parameters(var_list=discriminator_vars)))

        logger.info('Total Trainable Discriminator Parameters: {}'.format(
            tfutil.count_trainable_parameters(var_list=trainable_discriminator_vars)))

        saver.restore(session, restore_path)

        nb_instances = sentence1.shape[0]
        batches = make_batches(size=nb_instances, batch_size=batch_size)

        order = rs.permutation(nb_instances)

        sentences1 = sentence1[order]
        sentences2 = sentence2[order]

        sizes1 = sentence1_length[order]
        sizes2 = sentence2_length[order]

        labels = label[order]

        a_predictions_int_value = []
        b_predictions_int_value = []

        for batch_idx, (batch_start, batch_end) in tqdm(list(enumerate(batches))):
            batch_sentences1 = sentences1[batch_start:batch_end]
            batch_sentences2 = sentences2[batch_start:batch_end]
            batch_sizes1 = sizes1[batch_start:batch_end]
            batch_sizes2 = sizes2[batch_start:batch_end]

            batch_a_feed_dict = {
                sentence1_ph: batch_sentences1,
                sentence1_len_ph: batch_sizes1,

                sentence2_ph: batch_sentences2,
                sentence2_len_ph: batch_sizes2,

                dropout_keep_prob_ph: 1.0
            }

            batch_a_predictions_int_value = session.run(predictions_int, feed_dict=batch_a_feed_dict)
            a_predictions_int_value += batch_a_predictions_int_value.tolist()

            batch_b_feed_dict = {
                sentence1_ph: batch_sentences2,
                sentence1_len_ph: batch_sizes2,

                sentence2_ph: batch_sentences1,
                sentence2_len_ph: batch_sizes1,

                dropout_keep_prob_ph: 1.0
            }

            batch_b_predictions_int_value = session.run(predictions_int, feed_dict=batch_b_feed_dict)
            b_predictions_int_value += batch_b_predictions_int_value.tolist()

        logger.info('Number of examples: {}'.format(labels.shape[0]))

        train_accuracy_value = np.mean(labels == np.array(a_predictions_int_value))
        logger.info('Accuracy: {}'.format(train_accuracy_value))

        s1s2_con = (np.array(a_predictions_int_value) == contradiction_idx)
        s2s1_con = (np.array(b_predictions_int_value) == contradiction_idx)

        assert s1s2_con.shape == s2s1_con.shape

        s1s2_ent = (np.array(a_predictions_int_value) == entailment_idx)
        s2s1_ent = (np.array(b_predictions_int_value) == entailment_idx)

        s1s2_neu = (np.array(a_predictions_int_value) == neutral_idx)
        s2s1_neu = (np.array(b_predictions_int_value) == neutral_idx)

        logger.info('(S1 contradicts S2) XOR (S2 contradicts S1): {}'
                    .format(np.logical_xor(s1s2_con, s2s1_con).sum()))

        logger.info('(S1 contradicts S2): {}'
                    .format(s1s2_con.sum()))
        logger.info('(S1 contradicts S2) AND NOT(S2 contradicts S1)'
                    .format(np.logical_and(s1s2_con, np.logical_not(s2s1_con)).sum()))

        logger.info('(S1 entailment S2): {}'
                    .format(s1s2_ent.sum()))
        logger.info('(S1 entailment S2) AND (S2 contradicts S1): {}'
                    .format(np.logical_and(s1s2_ent, s2s1_con).sum()))

        logger.info('(S1 neutral S2): {}'
                    .format(s1s2_con.sum()))
        logger.info('(S1 neutral S2) AND (S2 contradicts S1): {}'
                    .format(np.logical_and(s1s2_neu, s2s1_con).sum()))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
