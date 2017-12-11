#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Running:
#  $ python3 ./bin/nli-dsearch-cli.py --has-bos --has-unk --restore models/snli/dam_1/dam_1

import argparse

import sys

import json
import pickle

import numpy as np
import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

from inferbeddings.models.training.util import make_batches

from inferbeddings.nli import util, tfutil
from inferbeddings.nli import ConditionalBiLSTM
from inferbeddings.nli import FeedForwardDAM
from inferbeddings.nli import FeedForwardDAMP
from inferbeddings.nli import FeedForwardDAMS
from inferbeddings.nli import ESIMv1

import logging

np.set_printoptions(threshold=np.nan)

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


def relu(x):
    return np.maximum(x, 0)


def log_perplexity(sentences,   sizes):
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
        feed = {
            lm_input_data_ph: x, lm_targets_ph: y, lm_initial_state: state
        }
        loss_value, state = session.run([lm_loss, lm_final_state], feed_dict=feed)
        loss_values += [loss_value]
    loss_values = np.array(loss_values).transpose()
    __sizes = _sizes - 2
    res = np.array([np.sum(loss_values[_i, :__sizes[_i]]) for _i in range(loss_values.shape[0])])
    return res


def inference(sentences1, sizes1, sentences2, sizes2):
    feed_dict = {
        sentence1_ph: sentences1, sentence1_len_ph: sizes1,
        sentence2_ph: sentences2, sentence2_len_ph: sizes2,
        dropout_keep_prob_ph: 1.0
    }
    probabilities_value = session.run(probabilities, feed_dict=feed_dict)

    def to_dict(x):
        return {'contradiction': x[contradiction_idx], 'neutral': x[neutral_idx], 'entailment': x[entailment_idx]}

    return [to_dict(probabilities_value[i, :]) for i in range(probabilities_value.shape[0])]


def contradiction_loss(sentences1, sizes1, sentences2, sizes2):
    feed_dict_1 = {
        sentence1_ph: sentences1, sentence1_len_ph: sizes1,
        sentence2_ph: sentences2, sentence2_len_ph: sizes2,
        dropout_keep_prob_ph: 1.0
    }
    feed_dict_2 = {
        sentence1_ph: sentences2, sentence1_len_ph: sizes2,
        sentence2_ph: sentences1, sentence2_len_ph: sizes1,
        dropout_keep_prob_ph: 1.0
    }
    probabilities_1 = session.run(probabilities, feed_dict=feed_dict_1)
    probabilities_2 = session.run(probabilities, feed_dict=feed_dict_2)
    ans_1 = probabilities_1[:, contradiction_idx]
    ans_2 = probabilities_2[:, contradiction_idx]
    res = relu(ans_1 - ans_2)
    return res


def entailment_loss(sentences1, sizes1, sentences2, sizes2):
    feed_dict_1 = {
        sentence1_ph: sentences1, sentence1_len_ph: sizes1,
        sentence2_ph: sentences2, sentence2_len_ph: sizes2,
        dropout_keep_prob_ph: 1.0
    }
    feed_dict_2 = {
        sentence1_ph: sentences2, sentence1_len_ph: sizes2,
        sentence2_ph: sentences1, sentence2_len_ph: sizes1,
        dropout_keep_prob_ph: 1.0
    }
    probabilities_1 = session.run(probabilities, feed_dict=feed_dict_1)
    probabilities_2 = session.run(probabilities, feed_dict=feed_dict_2)
    ans_1 = probabilities_1[:, entailment_idx]
    ans_2 = 1.0 - probabilities_2[:, contradiction_idx]
    res = relu(ans_1 - ans_2)
    return res


def neutral_loss(sentences1, sizes1, sentences2, sizes2):
    feed_dict_1 = {
        sentence1_ph: sentences1, sentence1_len_ph: sizes1,
        sentence2_ph: sentences2, sentence2_len_ph: sizes2,
        dropout_keep_prob_ph: 1.0
    }
    feed_dict_2 = {
        sentence1_ph: sentences2, sentence1_len_ph: sizes2,
        sentence2_ph: sentences1, sentence2_len_ph: sizes1,
        dropout_keep_prob_ph: 1.0
    }
    probabilities_1 = session.run(probabilities, feed_dict=feed_dict_1)
    probabilities_2 = session.run(probabilities, feed_dict=feed_dict_2)
    ans_1 = probabilities_1[:, neutral_idx]
    ans_2 = 1.0 - probabilities_2[:, contradiction_idx]
    res = relu(ans_1 - ans_2)
    return res


def loss(sentences1, sizes1, sentences2, sizes2,
         lambda_w=0.1, inconsistency_loss=contradiction_loss):
    inconsistency_loss_value = inconsistency_loss(sentences1=sentences1, sizes1=sizes1,
                                                  sentences2=sentences2, sizes2=sizes2)

    log_perplexity_1_value = log_perplexity(sentences=sentences1, sizes=sizes1)
    log_perplexity_2_value = log_perplexity(sentences=sentences2, sizes=sizes2)

    log_perplexity_value = log_perplexity_1_value + log_perplexity_2_value

    loss_value = inconsistency_loss_value - lambda_w * log_perplexity_value

    return loss_value, inconsistency_loss_value, log_perplexity_value


def corrupt(sentence1, size1, sentence2, size2,
            nb_corruptions=1024, nb_words=512):
    corruptions1 = np.repeat(a=[sentence1], repeats=nb_corruptions, axis=0)
    corruptions2 = np.repeat(a=[sentence2], repeats=nb_corruptions, axis=0)
    assert corruptions1.shape == (nb_corruptions, sentence1.shape[0])

    sizes1 = np.repeat(a=size1, repeats=nb_corruptions, axis=0)
    sizes2 = np.repeat(a=size2, repeats=nb_corruptions, axis=0)
    assert sizes1.shape[0] == corruptions1.shape[0]

    # Corrupt corruptions2
    for i in range(nb_corruptions):
        # Do not corrupt the last token - usually a '.' - corresponding to high=sizes2[i]
        where_to_corrupt = rs.randint(low=1, high=sizes2[i] - 1)
        new_word = rs.randint(low=1, high=nb_words)
        corruptions2[i, where_to_corrupt] = new_word

    return corruptions1, sizes1, corruptions2, sizes2


def search(sentences1, sizes1, sentences2, sizes2,
           lambda_w=0.1, inconsistency_loss=contradiction_loss,
           epsilon=1e-4, batch_size=32,
           nb_corruptions=1024, nb_words=256):

    loss_value, iloss_value, logperp_value = loss(sentences1=sentences1, sizes1=sizes1,
                                                  sentences2=sentences2, sizes2=sizes2,
                                                  lambda_w=lambda_w, inconsistency_loss=inconsistency_loss)

    # Find examples that have a nearly-zero inconsistency loss, and only work on making those more "adversarial"
    low_iloss_idxs = np.where(iloss_value < 1e-6)[0]

    for low_iloss_idx in low_iloss_idxs.tolist():
        sentence1, size1 = sentences1[low_iloss_idx, :], sizes1[low_iloss_idx]
        sentence2, size2 = sentences2[low_iloss_idx, :], sizes2[low_iloss_idx]

        sample_loss_value, sample_iloss_value, sample_logperp_value = \
            loss_value[low_iloss_idx], iloss_value[low_iloss_idx], logperp_value[low_iloss_idx]

        sentence1_str = ' '.join([index_to_token[tidx] for tidx in sentence1 if tidx != 0])
        sentence2_str = ' '.join([index_to_token[tidx] for tidx in sentence2 if tidx != 0])

        logger.info('SENTENCE 1 (inconsistency loss: {} / log-perplexity: {}): {}'
                    .format(sample_iloss_value, sample_logperp_value, sentence1_str))
        logger.info('SENTENCE 2 (inconsistency loss: {} / log-perplexity: {}): {}'
                    .format(sample_iloss_value, sample_logperp_value, sentence2_str))

        # Generate mutations that do not increase the perplexity too much, and maximise the inconsistency loss
        corruptions1, corruption_sizes1, corruptions2, corruption_sizes2 = \
            corrupt(sentence1=sentence1, size1=size1, sentence2=sentence2, size2=size2,
                    nb_corruptions=nb_corruptions, nb_words=nb_words)

        # Compute all relevant metrics for the corruptions
        nb_corruptions = corruptions1.shape[0]
        batches = make_batches(size=nb_corruptions, batch_size=batch_size)

        corruption_loss_values, corruption_iloss_values, corruption_logperp_values = [], [], []
        for batch_start, batch_end in batches:
            batch_corruptions1 = corruptions1[batch_start:batch_end, :]
            batch_corruption_sizes1 = corruption_sizes1[batch_start:batch_end]

            batch_corruptions2 = corruptions2[batch_start:batch_end, :]
            batch_corruption_sizes2 = corruption_sizes2[batch_start:batch_end]

            batch_loss_values, batch_iloss_values, batch_logperp_values = \
                loss(sentences1=batch_corruptions1, sizes1=batch_corruption_sizes1,
                     sentences2=batch_corruptions2, sizes2=batch_corruption_sizes2,
                     lambda_w=lambda_w, inconsistency_loss=inconsistency_loss)

            corruption_loss_values += batch_loss_values.tolist()
            corruption_iloss_values += batch_iloss_values.tolist()
            corruption_logperp_values += batch_logperp_values.tolist()

        corruption_loss_values = np.array(corruption_loss_values)
        corruption_iloss_values = np.array(corruption_iloss_values)
        corruption_logperp_values = np.array(corruption_logperp_values)

        # Sort the corruptions by their inconsistency loss:
        corruptions_order = np.argsort(corruption_iloss_values)[::-1]

        # Select corruptions that did not increase the log-perplexity too much
        low_perplexity_mask = corruption_logperp_values <= logperp_value[low_iloss_idx] + epsilon

        counter = 0
        for idx in corruptions_order.tolist():
            if idx in np.where(low_perplexity_mask)[0].tolist():
                if counter < 10:
                    corruption_str = ' '.join([index_to_token[tidx] for tidx in corruptions2[idx] if tidx != 0])
                    msg = '[{}] CORRUPTION 2 (inconsistency loss: {} / log-perplexity: {}): {}'\
                        .format(counter, corruption_iloss_values[idx], corruption_logperp_values[idx], corruption_str)

                    logger.info(msg)

                    _sentence1 = np.array([sentence1])
                    _size1 = np.array([size1])

                    _sentence2 = corruptions2[idx]
                    _size2 = np.array([size2])

                    probabilities_1 = inference(_sentence1, _size1, _sentence2, _size2)
                    probabilities_2 = inference(_sentence2, _size2, _sentence1, _size1)

                    msg = 'A -> B: {}\tB -> A: {}'.format(str(probabilities_1), str(probabilities_2))

                    logger.info(msg)
                counter += 1

    return


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
    argparser.add_argument('--lm', action='store', type=str, default='models/lm/')

    argparser.add_argument('--corrupt', '-c', action='store_true', default=False,
                           help='Corrupt examples so to maximise their inconsistency')
    argparser.add_argument('--most-violating', '-M', action='store_true', default=False,
                           help='Show most violating examples')

    argparser.add_argument('--epsilon', '-e', action='store', type=float, default=1e-4)
    argparser.add_argument('--lambda-weight', '-L', action='store', type=float, default=1.0)

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

    is_corrupt = args.corrupt
    is_most_violating = args.most_violating

    epsilon = args.epsilon
    lambda_w = args.lambda_weight

    np.random.seed(seed)
    tf.set_random_seed(seed)

    logger.debug('Reading corpus ..')
    data_is, _, _ = util.SNLI.generate(train_path=data_path, valid_path=None, test_path=None, is_lower=is_lower)
    logger.info('Data size: {}'.format(len(data_is)))

    # Enumeration of tokens start at index=3:
    # index=0 PADDING, index=1 START_OF_SENTENCE, index=2 END_OF_SENTENCE, index=3 UNKNOWN_WORD
    bos_idx, eos_idx, unk_idx = 1, 2, 3

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

    label_to_index = {
        'entailment': entailment_idx,
        'neutral': neutral_idx,
        'contradiction': contradiction_idx,
    }

    max_len = None

    args = dict(
        has_bos=has_bos, has_eos=has_eos, has_unk=has_unk,
        bos_idx=bos_idx, eos_idx=eos_idx, unk_idx=unk_idx,
        max_len=max_len)

    dataset = util.instances_to_dataset(data_is, token_to_index, label_to_index, **args)

    sentence1, sentence1_length = dataset['sentence1'], dataset['sentence1_length']
    sentence2, sentence2_length = dataset['sentence2'], dataset['sentence2_length']
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
        lm_initial_state = lm_cell.zero_state(lm_batch_size, tf.float32, )

        with tf.variable_scope('rnnlm'):
            lm_W = tf.get_variable(name='W', shape=[rnn_size, vocab_size],
                                   initializer=tf.contrib.layers.xavier_initializer())

            lm_b = tf.get_variable(name='b', shape=[vocab_size],
                                   initializer=tf.zeros_initializer())

            lm_emb_lookup = tf.nn.embedding_lookup(embedding_layer, lm_input_data_ph)
            lm_emb_projection = tf.contrib.layers.fully_connected(inputs=lm_emb_lookup, num_outputs=rnn_size,
                                                                  weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                                  biases_initializer=tf.zeros_initializer())

            lm_inputs = tf.split(lm_emb_projection, seq_length, 1)
            lm_inputs = [tf.squeeze(input_, [1]) for input_ in lm_inputs]

        lm_outputs, lm_last_state = legacy_seq2seq.rnn_decoder(decoder_inputs=lm_inputs, initial_state=lm_initial_state,
                                                               cell=lm_cell, loop_function=None, scope='rnnlm')

        lm_output = tf.reshape(tf.concat(lm_outputs, 1), [-1, rnn_size])

        lm_logits = tf.matmul(lm_output, lm_W) + lm_b
        lm_probabilities = tf.nn.softmax(lm_logits)

        global lm_loss, lm_cost, lm_final_state
        lm_loss = legacy_seq2seq.sequence_loss_by_example(logits=[lm_logits], targets=[tf.reshape(lm_targets_ph, [-1])],
                                                          weights=[tf.ones([lm_batch_size * seq_length])])
        lm_cost = tf.reduce_sum(lm_loss) / lm_batch_size / seq_length
        lm_final_state = lm_last_state

    discriminator_vars = tfutil.get_variables_in_scope(discriminator_scope_name)
    lm_vars = tfutil.get_variables_in_scope(lm_scope_name)

    predictions_int = tf.cast(predictions, tf.int32)

    saver = tf.train.Saver(discriminator_vars, max_to_keep=1)
    lm_saver = tf.train.Saver(lm_vars, max_to_keep=1)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    global session
    with tf.Session(config=session_config) as session:
        logger.info('Total Parameters: {}'.format(tfutil.count_trainable_parameters()))

        saver.restore(session, restore_path)

        lm_ckpt = tf.train.get_checkpoint_state(lm_path)
        lm_saver.restore(session, lm_ckpt.model_checkpoint_path)

        nb_instances = sentence1.shape[0]
        batches = make_batches(size=nb_instances, batch_size=batch_size)

        order = np.arange(nb_instances)

        sentences1 = sentence1[order]
        sentences2 = sentence2[order]

        sizes1 = sentence1_length[order]
        sizes2 = sentence2_length[order]

        labels = label[order]

        logger.info('Number of examples: {}'.format(labels.shape[0]))

        predictions_int_value = []
        c_losses, e_losses, n_losses = [], [], []

        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            batch_sentences1 = sentences1[batch_start:batch_end]
            batch_sentences2 = sentences2[batch_start:batch_end]

            batch_sizes1 = sizes1[batch_start:batch_end]
            batch_sizes2 = sizes2[batch_start:batch_end]

            batch_feed_dict = {
                sentence1_ph: batch_sentences1, sentence1_len_ph: batch_sizes1,
                sentence2_ph: batch_sentences2, sentence2_len_ph: batch_sizes2,
                dropout_keep_prob_ph: 1.0
            }

            batch_predictions_int = session.run(predictions_int, feed_dict=batch_feed_dict)
            predictions_int_value += batch_predictions_int.tolist()

            batch_c_loss = contradiction_loss(batch_sentences1, batch_sizes1, batch_sentences2, batch_sizes2)
            c_losses += batch_c_loss.tolist()

            batch_e_loss = entailment_loss(batch_sentences1, batch_sizes1, batch_sentences2, batch_sizes2)
            e_losses += batch_e_loss.tolist()

            batch_n_loss = neutral_loss(batch_sentences1, batch_sizes1, batch_sentences2, batch_sizes2)
            n_losses += batch_n_loss.tolist()

            if is_corrupt:
                search(sentences1=batch_sentences1, sizes1=batch_sizes1,
                       sentences2=batch_sentences2, sizes2=batch_sizes2,
                       batch_size=batch_size, epsilon=epsilon, lambda_w=lambda_w)
                # sys.exit(0)

        train_accuracy_value = np.mean(labels == np.array(predictions_int_value))
        logger.info('Accuracy: {0:.4f}'.format(train_accuracy_value))

        if is_most_violating:
            c_ranking = np.argsort(np.array(c_losses))[::-1]
            assert c_ranking.shape[0] == len(data_is)

            for i in range(min(1024, c_ranking.shape[0])):
                idx = c_ranking[i]
                print('[C/{}/{}] {} ({})'.format(i, idx, data_is[idx]['sentence1'], c_losses[idx]))
                print('[C/{}/{}] {} ({})'.format(i, idx, data_is[idx]['sentence2'], c_losses[idx]))

            e_ranking = np.argsort(np.array(e_losses))[::-1]
            assert e_ranking.shape[0] == len(data_is)

            for i in range(min(1024, e_ranking.shape[0])):
                idx = e_ranking[i]
                print('[E/{}/{}] {} ({})'.format(i, idx, data_is[idx]['sentence1'], e_losses[idx]))
                print('[E/{}/{}] {} ({})'.format(i, idx, data_is[idx]['sentence2'], e_losses[idx]))

            n_ranking = np.argsort(np.array(n_losses))[::-1]
            assert n_ranking.shape[0] == len(data_is)

            for i in range(min(1024, n_ranking.shape[0])):
                idx = n_ranking[i]
                print('[N/{}/{}] {} ({})'.format(i, idx, data_is[idx]['sentence1'], n_losses[idx]))
                print('[N/{}/{}] {} ({})'.format(i, idx, data_is[idx]['sentence2'], n_losses[idx]))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
