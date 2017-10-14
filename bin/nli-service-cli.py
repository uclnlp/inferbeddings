#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import os
import sys

import pickle

import tensorflow as tf

import nltk

from flask import Flask, request, jsonify
from flask.views import View

from inferbeddings.nli import ConditionalBiLSTM
from inferbeddings.nli import FeedForwardDAM
from inferbeddings.nli import FeedForwardDAMP
from inferbeddings.nli import FeedForwardDAMS
from inferbeddings.nli import ESIMv1

from werkzeug.serving import WSGIRequestHandler, BaseWSGIServer

import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(os.path.basename(sys.argv[0]))

WSGIRequestHandler.protocol_version = "HTTP/1.1"
BaseWSGIServer.protocol_version = "HTTP/1.1"

app = Flask('nli-service')


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

    argparser = argparse.ArgumentParser('NLI Service', formatter_class=formatter)

    argparser.add_argument('--model', '-m', action='store', type=str, default='cbilstm',
                           choices=['cbilstm', 'ff-dam', 'ff-damp', 'ff-dams', 'esim1'])

    argparser.add_argument('--embedding-size', '-e', action='store', type=int, default=300)
    argparser.add_argument('--representation-size', '-r', action='store', type=int, default=200)

    argparser.add_argument('--has-bos', action='store_true', default=False, help='Has <Beginning Of Sentence> token')
    argparser.add_argument('--has-eos', action='store_true', default=False, help='Has <End Of Sentence> token')
    argparser.add_argument('--has-unk', action='store_true', default=False, help='Has <Unknown Word> token')
    argparser.add_argument('--lower', '-l', action='store_true', default=False, help='Lowercase the corpus')

    argparser.add_argument('--fixed-embeddings', '-f', action='store_true')

    argparser.add_argument('--restore', action='store', type=str, default=None)

    args = argparser.parse_args(argv)

    model_name = args.model

    embedding_size = args.embedding_size
    representation_size = args.representation_size

    has_bos = args.has_bos
    has_eos = args.has_eos
    has_unk = args.has_unk
    is_lower = args.lower

    is_fixed_embeddings = args.fixed_embeddings

    restore_path = args.restore

    assert restore_path is not None

    with open('{}/index_to_token.p'.format(restore_path), 'rb') as f:
        index_to_token = pickle.load(f)

    token_to_index = {token: index for index, token in index_to_token.items()}

    bos_idx, eos_idx, unk_idx = 1, 2, 3

    entailment_idx, neutral_idx, contradiction_idx = 0, 1, 2
    vocab_size = max(token_to_index.values()) + 1

    sentence1_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence1')
    sentence2_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence2')

    sentence1_length_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence1_length')
    sentence2_length_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence2_length')

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

    tokenizer = nltk.tokenize.TreebankWordTokenizer()

    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, restore_path)

        class Service(View):
            methods = ['GET', 'POST']

            def dispatch_request(self):
                sentence1 = request.form['sentence1'] if 'sentence1' in request.form else request.args.get('sentence1')
                sentence2 = request.form['sentence2'] if 'sentence2' in request.form else request.args.get('sentence2')

                sentence1_tkns = tokenizer.tokenize(sentence1)
                sentence2_tkns = tokenizer.tokenize(sentence2)

                sentence1_ids = []
                sentence2_ids = []

                if has_bos:
                    sentence1_ids += [bos_idx]
                    sentence2_ids += [bos_idx]

                for token in sentence1_tkns:
                    if token in token_to_index:
                        sentence1_ids += [token_to_index[token]]
                    elif has_unk:
                        sentence1_ids += [unk_idx]

                for token in sentence2_tkns:
                    if token in token_to_index:
                        sentence2_ids += [token_to_index[token]]
                    elif has_unk:
                        sentence2_ids += [unk_idx]

                if has_eos:
                    sentence1_ids += [eos_idx]
                    sentence2_ids += [eos_idx]

                # Compute answer
                feed_dict = {
                    sentence1_ph: sentence1_ids,
                    sentence2_ph: sentence2_ids,
                    sentence1_length_ph: [len(sentence1_ids)],
                    sentence2_length_ph: [len(sentence2_ids)],
                    dropout_keep_prob_ph: 1.0
                }

                predictions = session.run(tf.nn.softmax(model.logits), feed_dict=feed_dict)[0]
                answer = {
                    'neutral': str(predictions[neutral_idx]),
                    'contradiction': str(predictions[contradiction_idx]),
                    'entailment': str(predictions[entailment_idx])
                }

                return jsonify(answer)

        app.add_url_rule('/v1/nli', view_func=Service.as_view('request'))

        app.run(host='0.0.0.0', port=8889, debug=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
