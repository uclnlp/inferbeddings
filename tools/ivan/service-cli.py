#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import os
import sys

import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify
from flask.views import View

from inferbeddings.rte import ConditionalBiLSTM
from inferbeddings.rte.dam import SimpleDAM, FeedForwardDAM, DAMP
from inferbeddings.rte.util import SNLI, count_trainable_parameters, train_tokenizer_on_instances

from inferbeddings.models.training import constraints

import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(os.path.basename(sys.argv[0]))

app = Flask('jack-the-service')

class InvalidAPIUsage(Exception):
    """
    Class used for handling error messages.
    """
    DEFAULT_STATUS_CODE = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        self.status_code = self.DEFAULT_STATUS_CODE
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


@app.errorhandler(InvalidAPIUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Regularising RTE via Adversarial Sets Regularisation',
                                        formatter_class=formatter)

    argparser.add_argument('--train', '-t', action='store', type=str, default='data/snli/snli_1.0_train.jsonl.gz')
    argparser.add_argument('--valid', '-v', action='store', type=str, default='data/snli/snli_1.0_dev.jsonl.gz')
    argparser.add_argument('--test', '-T', action='store', type=str, default='data/snli/snli_1.0_test.jsonl.gz')

    argparser.add_argument('--model', '-m', action='store', type=str, default='cbilstm',
                           choices=['cbilstm', 'simple-dam', 'ff-dam', 'damp'])
    argparser.add_argument('--optimizer', '-o', action='store', type=str, default='adagrad',
                           choices=['adagrad', 'adam'])

    argparser.add_argument('--embedding-size', action='store', type=int, default=300)
    argparser.add_argument('--representation-size', action='store', type=int, default=200)
    argparser.add_argument('--hidden-size', action='store', type=int, default=200)

    argparser.add_argument('--batch-size', action='store', type=int, default=1024)
    argparser.add_argument('--nb-epochs', action='store', type=int, default=1000)
    argparser.add_argument('--dropout-keep-prob', action='store', type=float, default=1.0)
    argparser.add_argument('--learning-rate', action='store', type=float, default=0.1)
    argparser.add_argument('--seed', action='store', type=int, default=0)

    argparser.add_argument('--semi-sort', action='store_true')
    argparser.add_argument('--fixed-embeddings', '-f', action='store_true')
    argparser.add_argument('--normalized-embeddings', '-n', action='store_true')
    argparser.add_argument('--use-masking', action='store_true')
    argparser.add_argument('--prepend-null-token', action='store_true')

    argparser.add_argument('--restore', action='store', type=str, default=None)

    args = argparser.parse_args(argv)

    train_path, valid_path, test_path = args.train, args.valid, args.test

    model_name = args.model
    optimizer_name = args.optimizer

    embedding_size = args.embedding_size
    representation_size = args.representation_size
    hidden_size = args.hidden_size

    dropout_keep_prob = args.dropout_keep_prob
    learning_rate = args.learning_rate
    seed = args.seed

    is_fixed_embeddings = args.fixed_embeddings
    is_normalized_embeddings = args.normalized_embeddings
    use_masking = args.use_masking
    prepend_null_token = args.prepend_null_token

    restore_path = args.restore

    np.random.seed(seed)
    tf.set_random_seed(seed)

    logger.debug('Reading corpus ..')
    train_instances, dev_instances, test_instances = SNLI.generate(
        train_path=train_path, valid_path=valid_path, test_path=test_path)

    logger.info('Train size: {}\tDev size: {}\tTest size: {}'.format(len(train_instances), len(dev_instances), len(test_instances)))

    logger.debug('Parsing corpus ..')

    num_words = None
    all_instances = train_instances + dev_instances + test_instances
    qs_tokenizer, a_tokenizer = train_tokenizer_on_instances(all_instances, num_words=num_words)

    neutral_idx = a_tokenizer.word_index['neutral'] - 1
    entailment_idx = a_tokenizer.word_index['entailment'] - 1
    contradiction_idx = a_tokenizer.word_index['contradiction'] - 1

    vocab_size = qs_tokenizer.num_words if qs_tokenizer.num_words else len(qs_tokenizer.word_index) + 1

    optimizer_class = None
    if optimizer_name == 'adagrad':
        optimizer_class = tf.train.AdagradOptimizer
    elif optimizer_name == 'adam':
        optimizer_class = tf.train.AdamOptimizer
    assert optimizer_class is not None

    optimizer = optimizer_class(learning_rate=learning_rate)

    model_kwargs = dict(optimizer=optimizer, vocab_size=vocab_size, embedding_size=embedding_size,
                        l2_lambda=None, trainable_embeddings=not is_fixed_embeddings)

    RTEModel = None
    if model_name == 'cbilstm':
        cbilstm_kwargs = dict(hidden_size=hidden_size,
                              dropout_keep_prob=dropout_keep_prob)
        model_kwargs.update(cbilstm_kwargs)
        RTEModel = ConditionalBiLSTM
    elif model_name == 'simple-dam':
        sd_kwargs = dict(use_masking=use_masking,
                         prepend_null_token=prepend_null_token)
        model_kwargs.update(sd_kwargs)
        RTEModel = SimpleDAM
    elif model_name == 'ff-dam':
        ff_kwargs = dict(representation_size=representation_size, dropout_keep_prob=dropout_keep_prob,
                         use_masking=use_masking, prepend_null_token=prepend_null_token)
        model_kwargs.update(ff_kwargs)
        RTEModel = FeedForwardDAM
    elif model_name == 'damp':
        damp_kwargs = dict(representation_size=representation_size, dropout_keep_prob=dropout_keep_prob,
                           use_masking=use_masking, prepend_null_token=prepend_null_token)
        model_kwargs.update(damp_kwargs)
        RTEModel = DAMP

    assert RTEModel is not None
    model = RTEModel(**model_kwargs)

    init_projection_steps = []
    learning_projection_steps = []

    if is_normalized_embeddings:
        unit_sphere_embeddings = constraints.unit_sphere(model.embeddings, norm=1.0)

        init_projection_steps += [unit_sphere_embeddings]
        if not is_fixed_embeddings:
            learning_projection_steps += [unit_sphere_embeddings]

        if prepend_null_token:
            unit_sphere_null_token = constraints.unit_sphere(model.null_token_embedding, norm=1.0)

            init_projection_steps += [unit_sphere_null_token]
            learning_projection_steps += [unit_sphere_null_token]

    saver = tf.train.Saver()

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    with tf.Session(config=session_config) as session:
        logger.debug('Total parameters: {}'.format(count_trainable_parameters()))
        saver.restore(session, restore_path)

        class Service(View):
            methods = ['GET', 'POST']

            def dispatch_request(self):
                sentence1 = request.args.get('sentence1')
                sentence2 = request.args.get('sentence2')

                if 'sentence1' in request.form:
                    sentence1 = request.form['sentence1']
                if 'sentence2' in request.form:
                    sentence2 = request.form['sentence2']

                sentence1_seq = qs_tokenizer.texts_to_sequences(sentence1)
                sentence2_seq = qs_tokenizer.texts_to_sequences(sentence2)

                sentence1_seq = [item for sublist in sentence1_seq for item in sublist]
                sentence2_seq = [item for sublist in sentence2_seq for item in sublist]

                # Compute answer
                feed_dict = {
                    model.sentence1: [sentence1_seq],
                    model.sentence2: [sentence2_seq],
                    model.sentence1_size: [len(sentence1_seq)],
                    model.sentence2_size: [len(sentence2_seq)]
                }

                print(feed_dict)

                predictions = session.run(model.logits, feed_dict=feed_dict)

                print(predictions)

                answer = {
                    'neutral': predictions[neutral_idx],
                    'contradiction': predictions[contradiction_idx],
                    'entailment': predictions[entailment_idx]
                }

                return jsonify(answer)

        app.add_url_rule('/v1/dam', view_func=Service.as_view('request'))

        app.run(host='0.0.0.0', port=8889, debug=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
