# -*- coding: utf-8 -*-

import pytest

import os
import sys

import numpy as np

from inferbeddings.io import iopen, load_glove, load_word2vec


def test_load_glove_large():
    glove_path = os.path.expanduser('~/data/glove/glove.840B.300d.txt')
    if os.path.isfile(glove_path):
        word_set = {'machine', 'learning'}
        with open(glove_path, 'r') as stream:
            word_to_embedding = load_glove(stream=stream, words=word_set)

        machine_vector = word_to_embedding['machine']
        learning_vector = word_to_embedding['learning']

        np.testing.assert_allclose(machine_vector[1:4], [-0.12538, 0.38888, 0.48011], rtol=1e-3)
        np.testing.assert_allclose(learning_vector[1:4], [0.047511, 0.1404, -0.11736], rtol=1e-3)


def test_load_word2vec():
    word2vec_path = os.path.expanduser('~/data/word2vec/GoogleNews-vectors-negative300.bin.gz')
    if os.path.isfile(word2vec_path):
        word_set = {'machine', 'learning'}
        with iopen('data/word2vec/GoogleNews-vectors-negative300.bin.gz') as stream:
            word_to_embedding = load_word2vec(stream=stream, words=word_set)

        machine_vector = word_to_embedding['machine']
        learning_vector = word_to_embedding['learning']

        print(machine_vector)
        print(learning_vector)


def test_load_glove():
    glove_path = 'data/glove/glove.6B.50d.txt.gz_X'
    if os.path.isfile(glove_path):
        word_set = {'house'}
        with iopen(glove_path, 'r') as stream:
            model = load_glove(stream=stream, words=word_set)
        assert 'house' in model
        assert 0.60136 < model['house'][0] < 0.60138


if __name__ == '__main__':
    pytest.main([__file__])
