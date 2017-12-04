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
#  $ python3 ./bin/nli-debug-cli.py --has-bos --has-unk --restore models/snli/dam_1/dam_1

def main(argv):
    logger.info('Command line: {}'.format(' '.join(arg for arg in argv)))

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
    argparser.add_argument('--lower', '-l', action='store_true', default=False, help='Lowercase the corpus')

    argparser.add_argument('--restore', action='store', type=str, default=None)

    argparser.add_argument('--check-transitivity', '-x', action='store_true', default=False)

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

    is_check_transitivity = args.check_transitivity

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
        embedding_layer = tf.get_variable('embeddings', shape=[vocab_size, embedding_size], trainable=False)

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
        probabilities = tf.nn.softmax(logits)

        predictions = tf.argmax(logits, axis=1, name='predictions')

    discriminator_vars = tfutil.get_variables_in_scope(discriminator_scope_name)

    trainable_discriminator_vars = list(discriminator_vars)

    predictions_int = tf.cast(predictions, tf.int32)

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

        a_probabilities_value = []
        b_probabilities_value = []

        for batch_idx, (batch_start, batch_end) in tqdm(list(enumerate(batches))):
            batch_sentences1 = sentences1[batch_start:batch_end]
            batch_sentences2 = sentences2[batch_start:batch_end]
            batch_sizes1 = sizes1[batch_start:batch_end]
            batch_sizes2 = sizes2[batch_start:batch_end]

            batch_a_feed_dict = {
                sentence1_ph: batch_sentences1, sentence1_len_ph: batch_sizes1,
                sentence2_ph: batch_sentences2, sentence2_len_ph: batch_sizes2,
                dropout_keep_prob_ph: 1.0
            }

            batch_a_predictions_int_value, batch_a_probabilities_value = session.run(
                [predictions_int, probabilities], feed_dict=batch_a_feed_dict)

            a_predictions_int_value += batch_a_predictions_int_value.tolist()
            for i in range(batch_a_probabilities_value.shape[0]):
                a_probabilities_value += [{
                    'neutral': batch_a_probabilities_value[i, neutral_idx],
                    'contradiction': batch_a_probabilities_value[i, contradiction_idx],
                    'entailment': batch_a_probabilities_value[i, entailment_idx]
                }]

            batch_b_feed_dict = {
                sentence1_ph: batch_sentences2, sentence1_len_ph: batch_sizes2,
                sentence2_ph: batch_sentences1, sentence2_len_ph: batch_sizes1,
                dropout_keep_prob_ph: 1.0
            }

            batch_b_predictions_int_value, batch_b_probabilities_value = session.run(
                [predictions_int, probabilities], feed_dict=batch_b_feed_dict)
            b_predictions_int_value += batch_b_predictions_int_value.tolist()
            for i in range(batch_b_probabilities_value.shape[0]):
                b_probabilities_value += [{
                    'neutral': batch_b_probabilities_value[i, neutral_idx],
                    'contradiction': batch_b_probabilities_value[i, contradiction_idx],
                    'entailment': batch_b_probabilities_value[i, entailment_idx]
                }]

        for i, instance in enumerate(data_is):
            instance.update({
                'a': a_probabilities_value[i],
                'b': b_probabilities_value[i],
            })

        logger.info('Number of examples: {}'.format(labels.shape[0]))

        train_accuracy_value = np.mean(labels == np.array(a_predictions_int_value))
        logger.info('Accuracy: {0:.4f}'.format(train_accuracy_value))

        s1s2_con = (np.array(a_predictions_int_value) == contradiction_idx)
        s2s1_con = (np.array(b_predictions_int_value) == contradiction_idx)

        assert s1s2_con.shape == s2s1_con.shape

        s1s2_ent = (np.array(a_predictions_int_value) == entailment_idx)
        s2s1_ent = (np.array(b_predictions_int_value) == entailment_idx)

        s1s2_neu = (np.array(a_predictions_int_value) == neutral_idx)
        s2s1_neu = (np.array(b_predictions_int_value) == neutral_idx)

        a = np.logical_xor(s1s2_con, s2s1_con)
        logger.info('(S1 contradicts S2) XOR (S2 contradicts S1): {0}'.format(a.sum()))

        b = s1s2_con
        logger.info('(S1 contradicts S2): {0}'.format(b.sum()))
        c = np.logical_and(s1s2_con, np.logical_not(s2s1_con))
        logger.info('(S1 contradicts S2) AND NOT(S2 contradicts S1): {0} ({1:.4f})'.format(c.sum(), c.sum() / b.sum()))

        with open('c.p', 'wb') as f:
            tmp = [data_is[i] for i in np.where(c)[0].tolist()]
            pickle.dump(tmp, f)

        d = s1s2_ent
        logger.info('(S1 entailment S2): {0}'.format(d.sum()))
        e = np.logical_and(s1s2_ent, s2s1_con)
        logger.info('(S1 entailment S2) AND (S2 contradicts S1): {0} ({1:.4f})'.format(e.sum(), e.sum() / d.sum()))

        with open('e.p', 'wb') as f:
            tmp = [data_is[i] for i in np.where(e)[0].tolist()]
            pickle.dump(tmp, f)

        f = s1s2_con
        logger.info('(S1 neutral S2): {0}'.format(f.sum()))
        g = np.logical_and(s1s2_neu, s2s1_con)
        logger.info('(S1 neutral S2) AND (S2 contradicts S1): {0} ({1:.4f})'.format(g.sum(), g.sum() / f.sum()))

        with open('g.p', 'wb') as f:
            tmp = [data_is[i] for i in np.where(g)[0].tolist()]
            pickle.dump(tmp, f)

        if is_check_transitivity:
            # Find S1, S2 such that entails(S1, S2)
            print(type(s1s2_ent))

            c_predictions_int_value = []
            c_probabilities_value = []

            d_predictions_int_value = []
            d_probabilities_value = []

            # Find candidate S3 sentences
            # order = rs.permutation(nb_instances)
            order = np.arange(nb_instances)

            sentences3 = sentence2[order]
            sizes3 = sentence2_length[order]

            for batch_idx, (batch_start, batch_end) in tqdm(list(enumerate(batches))):
                batch_sentences2 = sentences2[batch_start:batch_end]
                batch_sentences3 = sentences3[batch_start:batch_end]

                batch_sizes2 = sizes2[batch_start:batch_end]
                batch_sizes3 = sizes3[batch_start:batch_end]

                batch_c_feed_dict = {
                    sentence1_ph: batch_sentences2, sentence1_len_ph: batch_sizes2,
                    sentence2_ph: batch_sentences3, sentence2_len_ph: batch_sizes3,
                    dropout_keep_prob_ph: 1.0
                }

                batch_c_predictions_int_value, batch_c_probabilities_value = session.run(
                    [predictions_int, probabilities], feed_dict=batch_c_feed_dict)

                c_predictions_int_value += batch_c_predictions_int_value.tolist()
                for i in range(batch_c_probabilities_value.shape[0]):
                    c_probabilities_value += [{
                        'neutral': batch_c_probabilities_value[i, neutral_idx],
                        'contradiction': batch_c_probabilities_value[i, contradiction_idx],
                        'entailment': batch_c_probabilities_value[i, entailment_idx]
                    }]

                batch_sentences1 = sentences1[batch_start:batch_end]
                batch_sentences3 = sentences3[batch_start:batch_end]

                batch_sizes1 = sizes1[batch_start:batch_end]
                batch_sizes3 = sizes3[batch_start:batch_end]

                batch_d_feed_dict = {
                    sentence1_ph: batch_sentences1, sentence1_len_ph: batch_sizes1,
                    sentence2_ph: batch_sentences3, sentence2_len_ph: batch_sizes3,
                    dropout_keep_prob_ph: 1.0
                }

                batch_d_predictions_int_value, batch_d_probabilities_value = session.run(
                    [predictions_int, probabilities], feed_dict=batch_d_feed_dict)

                d_predictions_int_value += batch_d_predictions_int_value.tolist()
                for i in range(batch_d_probabilities_value.shape[0]):
                    d_probabilities_value += [{
                        'neutral': batch_d_probabilities_value[i, neutral_idx],
                        'contradiction': batch_d_probabilities_value[i, contradiction_idx],
                        'entailment': batch_d_probabilities_value[i, entailment_idx]
                    }]

            s2s3_ent = (np.array(c_predictions_int_value) == entailment_idx)
            s1s3_ent = (np.array(c_predictions_int_value) == entailment_idx)

            body = np.logical_and(s1s2_ent, s2s3_ent)
            body_not_head = np.logical_and(body, np.logical_not(s1s3_ent))

            logger.info('(S1 entails S2) and (S2 entails S3): {0}'.format(body.sum()))
            logger.info('body AND NOT(head): {0} ({1:.4f})'
                        .format(body_not_head.sum(), body_not_head.sum() / body.sum()))

            with open('h.p', 'wb') as f:
                tmp = []
                for idx in np.where(body_not_head)[0].tolist():
                    s1 = data_is[idx]['sentence1']
                    s2 = data_is[idx]['sentence2']
                    s3 = data_is[order[idx]]['sentence2']
                    tmp += [{
                        's1': s1, 's2': s2, 's3': s3
                    }]
                pickle.dump(tmp, f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
