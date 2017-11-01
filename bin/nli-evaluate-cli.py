#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import os
import sys

import json
import gzip
import pickle

import numpy as np
import tensorflow as tf

import nltk

from inferbeddings.nli import util, tfutil

from inferbeddings.nli import ConditionalBiLSTM
from inferbeddings.nli import FeedForwardDAM
from inferbeddings.nli import FeedForwardDAMP
from inferbeddings.nli import FeedForwardDAMS
from inferbeddings.nli import ESIMv1

import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('NLI Service', formatter_class=formatter)

    argparser.add_argument('--model', '-m', action='store', type=str, default='cbilstm',
                           choices=['cbilstm', 'ff-dam', 'ff-damp', 'ff-dams', 'esim1'])

    argparser.add_argument('--embedding-size', '-e', action='store', type=int, default=300)
    argparser.add_argument('--representation-size', '-r', action='store', type=int, default=200)

    argparser.add_argument('--has-bos', action='store_true', default=False, help='Has <Beginning Of Sentence> token')
    argparser.add_argument('--has-eos', action='store_true', default=False, help='Has <End Of Sentence> token')
    argparser.add_argument('--has-unk', action='store_true', default=False, help='Has <Unknown Word> token')
    argparser.add_argument('--lower', '-l', action='store_true', default=False, help='Lowercase the corpus')

    argparser.add_argument('--restore', '-R', action='store', type=str, default=None, required=True)

    argparser.add_argument('--eval', action='store', default=None, type=str)
    argparser.add_argument('--batch-size', '-b', action='store', default=32, type=int)

    args = argparser.parse_args(argv)

    model_name = args.model

    embedding_size = args.embedding_size
    representation_size = args.representation_size

    has_bos = args.has_bos
    has_eos = args.has_eos
    has_unk = args.has_unk
    is_lower = args.lower

    restore_path = args.restore

    eval_path = args.eval
    batch_size = args.batch_size

    with open('{}_index_to_token.p'.format(restore_path), 'rb') as f:
        index_to_token = pickle.load(f)

    token_to_index = {token: index for index, token in index_to_token.items()}

    # Enumeration of tokens start at index=3:
    # index=0 PADDING, index=1 START_OF_SENTENCE, index=2 END_OF_SENTENCE, index=3 UNKNOWN_WORD
    bos_idx, eos_idx, unk_idx = 1, 2, 3

    entailment_idx, neutral_idx, contradiction_idx = 0, 1, 2
    label_to_index = {
        'entailment': entailment_idx,
        'neutral': neutral_idx,
        'contradiction': contradiction_idx,
    }
    vocab_size = max(token_to_index.values()) + 1

    sentence1_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence1')
    sentence2_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence2')

    sentence1_len_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence1_length')
    sentence2_len_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence2_length')

    dropout_keep_prob_ph = tf.placeholder(tf.float32, name='dropout_keep_prob')

    clipped_sentence1 = tfutil.clip_sentence(sentence1_ph, sentence1_len_ph)
    clipped_sentence2 = tfutil.clip_sentence(sentence2_ph, sentence2_len_ph)

    discriminator_scope_name = 'discriminator'
    with tf.variable_scope(discriminator_scope_name):

        embedding_layer = tf.get_variable('embeddings', shape=[vocab_size, embedding_size])

        sentence1_embedding = tf.nn.embedding_lookup(embedding_layer, clipped_sentence1)
        sentence2_embedding = tf.nn.embedding_lookup(embedding_layer, clipped_sentence2)

        model_kwargs = dict(
            sequence1=sentence1_embedding, sequence1_length=sentence1_len_ph,
            sequence2=sentence2_embedding, sequence2_length=sentence2_len_ph,
            representation_size=representation_size, dropout_keep_prob=dropout_keep_prob_ph)

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
        predictions_op = tf.argmax(logits, axis=1, name='predictions')

    discriminator_vars = tfutil.get_variables_in_scope(discriminator_scope_name)

    sentence1_all = []
    sentence2_all = []
    gold_label_all = []

    with gzip.open(eval_path, 'rb') as f:
        for line in f:
            decoded_line = line.decode('utf-8')

            if is_lower:
                decoded_line = decoded_line.lower()

            obj = json.loads(decoded_line)

            gold_label = obj['gold_label']

            if gold_label in ['contradiction', 'entailment', 'neutral']:
                gold_label_all += [label_to_index[gold_label]]

                sentence1_parse = obj['sentence1_parse']
                sentence2_parse = obj['sentence2_parse']

                sentence1_tree = nltk.Tree.fromstring(sentence1_parse)
                sentence2_tree = nltk.Tree.fromstring(sentence2_parse)

                sentence1_tokens = sentence1_tree.leaves()
                sentence2_tokens = sentence2_tree.leaves()

                sentence1_ids = []
                sentence2_ids = []

                if has_bos:
                    sentence1_ids += [bos_idx]
                    sentence2_ids += [bos_idx]

                for token in sentence1_tokens:
                    if token in token_to_index:
                        sentence1_ids += [token_to_index[token]]
                    elif has_unk:
                        sentence1_ids += [unk_idx]

                for token in sentence2_tokens:
                    if token in token_to_index:
                        sentence2_ids += [token_to_index[token]]
                    elif has_unk:
                        sentence2_ids += [unk_idx]

                if has_eos:
                    sentence1_ids += [eos_idx]
                    sentence2_ids += [eos_idx]

                sentence1_all += [sentence1_ids]
                sentence2_all += [sentence2_ids]

    sentence1_all_len = [len(s) for s in sentence1_all]
    sentence2_all_len = [len(s) for s in sentence2_all]

    np_sentence1 = util.pad_sequences(sequences=sentence1_all)
    np_sentence2 = util.pad_sequences(sequences=sentence2_all)

    np_sentence1_len = np.array(sentence1_all_len)
    np_sentence2_len = np.array(sentence2_all_len)

    gold_label = np.array(gold_label_all)

    with tf.Session() as session:
        saver = tf.train.Saver(discriminator_vars, max_to_keep=1)
        saver.restore(session, restore_path)

        from inferbeddings.models.training.util import make_batches
        nb_instances = gold_label.shape[0]
        batches = make_batches(size=nb_instances, batch_size=batch_size)

        predictions = []

        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            feed_dict = {
                sentence1_ph: np_sentence1[batch_start:batch_end],
                sentence2_ph: np_sentence2[batch_start:batch_end],
                sentence1_len_ph: np_sentence1_len[batch_start:batch_end],
                sentence2_len_ph: np_sentence2_len[batch_start:batch_end],
                dropout_keep_prob_ph: 1.0
            }

            _predictions = session.run(predictions_op, feed_dict=feed_dict)
            predictions += _predictions.tolist()

        matches = np.array(predictions) == gold_label
        print(np.mean(matches))

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
