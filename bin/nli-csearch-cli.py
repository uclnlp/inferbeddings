#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Running:
#  $ python3 ./bin/nli-dsearch-cli.py --has-bos --has-unk --restore models/snli/dam_1/dam_1

import sys

import json
import pickle

import argparse

import numpy as np
import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

from inferbeddings.nli import util, tfutil
from inferbeddings.nli import ConditionalBiLSTM
from inferbeddings.nli import FeedForwardDAM
from inferbeddings.nli import FeedForwardDAMP
from inferbeddings.nli import FeedForwardDAMS
from inferbeddings.nli import ESIMv1

from inferbeddings.nli.regularizers.x import AdversarialSets
from inferbeddings.lm.decoder import decode

import logging

# np.set_printoptions(threshold=np.nan)

logger = logging.getLogger(__name__)
rs = np.random.RandomState(0)

entailment_idx, neutral_idx, contradiction_idx = 0, 1, 2

sentence1_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence1')
sentence2_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence2')

sentence1_len_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence1_length')
sentence2_len_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence2_length')

dropout_keep_prob_ph = tf.placeholder(tf.float32, name='dropout_keep_prob')

index_to_token = token_to_index = None
session = probabilities = None

lm_input_data_ph = lm_targets_ph = None
lm_cell = lm_initial_state = lm_final_state = None

lm_loss = lm_cost = None


def log_perplexity(sentences, sizes):
    assert sentences.shape[0] == sizes.shape[0]
    _batch_size = sentences.shape[0]
    x = np.zeros(shape=(_batch_size, 1))
    y = np.zeros(shape=(_batch_size, 1))
    _sentences, _sizes = sentences[:, 1:], sizes[:] - 1
    state = session.run(lm_cell.zero_state(_batch_size, tf.float32))
    loss_values = []
    for j in range(_sizes.max() - 1):
        x[:, 0] = _sentences[:, j]
        y[:, 0] = _sentences[:, j + 1]
        feed = {lm_input_data_ph: x, lm_targets_ph: y, lm_initial_state: state}
        loss_value, state = session.run([lm_loss, lm_final_state], feed_dict=feed)
        loss_values += [loss_value]
    loss_values = np.array(loss_values).transpose()
    __sizes = _sizes - 2
    res = np.array([np.sum(loss_values[_i, :__sizes[_i]]) for _i in range(loss_values.shape[0])])
    return res


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

    argparser.add_argument('--batch-size', '-b', action='store', type=int, default=32)
    argparser.add_argument('--seq-length', action='store', type=int, default=5)

    argparser.add_argument('--seed', action='store', type=int, default=0)

    argparser.add_argument('--has-bos', action='store_true', default=False, help='Has <Beginning Of Sentence> token')
    argparser.add_argument('--has-eos', action='store_true', default=False, help='Has <End Of Sentence> token')
    argparser.add_argument('--has-unk', action='store_true', default=False, help='Has <Unknown Word> token')
    argparser.add_argument('--lower', '-l', action='store_true', default=False, help='Lowercase the corpus')

    argparser.add_argument('--restore', action='store', type=str, default=None)
    argparser.add_argument('--lm', action='store', type=str, default='models/lm/')

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
    lm_path = args.lm

    np.random.seed(seed)
    tf.set_random_seed(seed)

    logger.debug('Reading corpus ..')
    data_is, _, _ = util.SNLI.generate(train_path=data_path, valid_path=None, test_path=None, is_lower=is_lower)
    logger.info('Data size: {}'.format(len(data_is)))

    # Enumeration of tokens start at index=3:
    # index=0 PADDING, index=1 START_OF_SENTENCE, index=2 END_OF_SENTENCE, index=3 UNKNOWN_WORD
    pad_idx, bos_idx, eos_idx, unk_idx = 0, 1, 2, 3

    global index_to_token, token_to_index
    with open('{}_index_to_token.p'.format(restore_path), 'rb') as f:
        index_to_token = pickle.load(f)

    index_to_token.update({0: '<PAD>', 1: '<BOS>', 2: '<UNK>'})

    token_to_index = {token: index for index, token in index_to_token.items()}

    with open('{}/config.json'.format(lm_path), 'r') as f:
        config = json.load(f)

    seq_length = 1
    lm_batch_size = batch_size
    rnn_size = config['rnn_size']
    num_layers = config['num_layers']

    label_to_index = {'entailment': entailment_idx, 'neutral': neutral_idx, 'contradiction': contradiction_idx}
    max_len = None

    args = dict(
        has_bos=has_bos, has_eos=has_eos, has_unk=has_unk,
        bos_idx=bos_idx, eos_idx=eos_idx, unk_idx=unk_idx,
        max_len=max_len)

    dataset = util.instances_to_dataset(data_is, token_to_index, label_to_index, **args)

    sentence1 = dataset['sentence1']
    sentence1_length = dataset['sentence1_length']

    sentence2 = dataset['sentence2'],
    sentence2_length = dataset['sentence2_length']

    label = dataset['label']

    clipped_sentence1 = tfutil.clip_sentence(sentence1_ph, sentence1_len_ph)
    clipped_sentence2 = tfutil.clip_sentence(sentence2_ph, sentence2_len_ph)

    vocab_size = max(token_to_index.values()) + 1

    discriminator_scope_name = 'discriminator'
    with tf.variable_scope(discriminator_scope_name):
        embedding_layer = tf.get_variable('embeddings', shape=[vocab_size, embedding_size], trainable=False)
        sentence1_embedding = tf.nn.embedding_lookup(embedding_layer, clipped_sentence1)
        sentence2_embedding = tf.nn.embedding_lookup(embedding_layer, clipped_sentence2)

        model_kwargs = dict(
            sequence1=sentence1_embedding, sequence1_length=sentence1_len_ph,
            sequence2=sentence2_embedding, sequence2_length=sentence2_len_ph,
            representation_size=representation_size,
            dropout_keep_prob=dropout_keep_prob_ph)

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

        global probabilities
        probabilities = tf.nn.softmax(logits)

        predictions = tf.argmax(logits, axis=1, name='predictions')

    lm_scope_name = 'language_model'
    with tf.variable_scope(lm_scope_name):
        cell_fn = rnn.BasicLSTMCell
        cells = [cell_fn(rnn_size) for _ in range(num_layers)]

        global lm_cell
        lm_cell = rnn.MultiRNNCell(cells)

        global lm_input_data_ph, lm_targets_ph, lm_initial_state
        lm_input_data_ph = tf.placeholder(tf.int32, [None, seq_length], name='input_data')
        lm_targets_ph = tf.placeholder(tf.int32, [None, seq_length], name='targets')

        lm_initial_state = lm_cell.zero_state(lm_batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            lm_W = tf.get_variable(name='W',
                                   shape=[rnn_size, vocab_size],
                                   initializer=tf.contrib.layers.xavier_initializer())

            lm_b = tf.get_variable(name='b',
                                   shape=[vocab_size],
                                   initializer=tf.zeros_initializer())

            lm_emb_lookup = tf.nn.embedding_lookup(embedding_layer, lm_input_data_ph)

            lm_emb_projection = tf.contrib.layers.fully_connected(inputs=lm_emb_lookup,
                                                                  num_outputs=rnn_size,
                                                                  weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                                  biases_initializer=tf.zeros_initializer())

            lm_inputs = tf.split(lm_emb_projection, seq_length, 1)
            lm_inputs = [tf.squeeze(input_, [1]) for input_ in lm_inputs]

        lm_outputs, lm_last_state = legacy_seq2seq.rnn_decoder(decoder_inputs=lm_inputs,
                                                               initial_state=lm_initial_state,
                                                               cell=lm_cell,
                                                               loop_function=None,
                                                               scope='rnnlm')

        lm_output = tf.reshape(tf.concat(lm_outputs, 1), [-1, rnn_size])

        lm_logits = tf.matmul(lm_output, lm_W) + lm_b
        lm_probabilities = tf.nn.softmax(lm_logits)

        global lm_loss, lm_cost, lm_final_state
        lm_loss = legacy_seq2seq.sequence_loss_by_example(logits=[lm_logits],
                                                          targets=[tf.reshape(lm_targets_ph, [-1])],
                                                          weights=[tf.ones([lm_batch_size * seq_length])])
        lm_cost = tf.reduce_sum(lm_loss) / lm_batch_size / seq_length
        lm_final_state = lm_last_state

    discriminator_vars = tfutil.get_variables_in_scope(discriminator_scope_name)
    lm_vars = tfutil.get_variables_in_scope(lm_scope_name)

    predictions_int = tf.cast(predictions, tf.int32)

    saver = tf.train.Saver(discriminator_vars, max_to_keep=1)
    lm_saver = tf.train.Saver(lm_vars, max_to_keep=1)

    a_batch_size = 1
    a_sequence_length = 16

    adversary_scope_name = discriminator_scope_name
    with tf.variable_scope(adversary_scope_name):
        adversary = AdversarialSets(model_class=model_class,
                                    model_kwargs=model_kwargs,
                                    embedding_size=embedding_size,
                                    scope_name='adversary',
                                    batch_size=a_batch_size,
                                    sequence_length=a_sequence_length,
                                    entailment_idx=entailment_idx,
                                    contradiction_idx=contradiction_idx,
                                    neutral_idx=neutral_idx)

        a_loss, a_sequence_lst = adversary.rule6_loss()
        a_sequence1_var, a_sequence2_var = a_sequence_lst

    a_optimizer_scope_name = 'adversary/optimizer'
    with tf.variable_scope(a_optimizer_scope_name):
        a_optimizer = tf.train.AdamOptimizer()
        a_step = a_optimizer.minimize(- a_loss, var_list=[a_sequence2_var])

    a_optimizer_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                         scope=a_optimizer_scope_name)
    a_optimizer_vars_init = tf.variables_initializer(a_optimizer_vars)

    a_var_to_assign_op = dict()
    a_var_value_ph = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name='a_var_value')
    for a_var in a_sequence_lst:
        a_var_to_assign_op[a_var] = a_var.assign(a_var_value_ph)
    a_init_op = tf.variables_initializer(a_sequence_lst)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    text1 = 'A man sleeps on the beach after removing his helmet'.split() + ['.']
    sentence1_ids = [token_to_index[token] for token in text1]

    text2 = 'A man sleeps by his bike'.split() + ['.']
    sentence2_ids = [token_to_index[token] for token in text2]

    global session
    with tf.Session(config=session_config) as session:
        logger.info('Total Parameters: {}'.format(tfutil.count_trainable_parameters()))

        saver.restore(session, restore_path)

        lm_ckpt = tf.train.get_checkpoint_state(lm_path)
        lm_saver.restore(session, lm_ckpt.model_checkpoint_path)

        emb_layer_value = session.run(embedding_layer)
        assert emb_layer_value.shape == (vocab_size, embedding_size)

        sentences, sizes = np.array([sentence1_ids]), np.array([len(sentence1_ids)])
        assert log_perplexity(sentences, sizes) >= 0.0

        logger.info('Initialising adversarial sequences ..')

        initial_sentence_emb_value = np.zeros(shape=(a_batch_size, a_sequence_length, embedding_size))

        initial_sentence_emb_value[0, 0, :] = emb_layer_value[bos_idx, :]
        for i in range(1, a_sequence_length):
            initial_sentence_emb_value[0, i, :] = emb_layer_value[pad_idx, :]

        sentence1_emb_value = initial_sentence_emb_value.copy()
        for i, idx in enumerate(sentence1_ids, start=1):
            sentence1_emb_value[0, i, :] = emb_layer_value[idx, :]

        sentence2_emb_value = initial_sentence_emb_value.copy()
        for i, idx in enumerate(sentence2_ids, start=1):
            sentence2_emb_value[0, i, :] = emb_layer_value[idx, :]

        assert len(a_sequence_lst) == 2
        sentence1_emb_var, sentence2_emb_var = a_sequence_lst[0], a_sequence_lst[1]

        session.run(a_init_op)

        feed = {a_var_value_ph: sentence1_emb_value}
        session.run(a_var_to_assign_op[sentence1_emb_var], feed_dict=feed)

        feed = {a_var_value_ph: sentence2_emb_value}
        session.run(a_var_to_assign_op[sentence2_emb_var], feed_dict=feed)

        session.run(a_optimizer_vars_init)

        a_word_idx = 3

        for e_idx in range(1024):
            a_loss_value = session.run(a_loss, feed_dict={dropout_keep_prob_ph: 1})

            d1_lst = decode(sequence_embedding=session.run(a_sequence1_var)[0, :, :],
                            embedding_matrix=emb_layer_value[:, :], index_to_token=index_to_token)

            d2_lst = decode(sequence_embedding=session.run(a_sequence2_var)[0, :, :],
                            embedding_matrix=emb_layer_value[:, :], index_to_token=index_to_token)

            print('{} {} {}'.format(e_idx, a_loss_value, ' '.join(d1_lst)))
            print('{} {} {}'.format(e_idx, a_loss_value, ' '.join(d2_lst)))

            session.run(a_step, feed_dict={dropout_keep_prob_ph: 1})

            # Clamping all word embeddings, except for the 3rd word
            u_sentence2_emb = sentence2_emb_value.copy()
            u_sentence2_emb[:, a_word_idx, :] = session.run(a_sequence2_var)[:, a_word_idx, :]

            # Normalising the embedding
            u_sentence2_emb /= np.linalg.norm(u_sentence2_emb, axis=2).reshape((batch_size, -1, 1))

            # Updating the embedding
            feed = {a_var_value_ph: u_sentence2_emb}
            session.run(a_var_to_assign_op[sentence2_emb_var], feed_dict=feed)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
