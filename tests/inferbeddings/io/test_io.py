# -*- coding: utf-8 -*-

import pytest

import os

import numpy as np

from inferbeddings.io import load_glove, load_word2vec


@pytest.mark.light
def test_load_glove_large():
    glove_path = os.path.expanduser('~/data/glove/glove.840B.300d.txt')
    if os.path.isfile(glove_path):
        word_set = {'machine', 'learning'}
        word_to_embedding = load_glove(path=glove_path, words=word_set)

        machine_vector = word_to_embedding['machine']
        learning_vector = word_to_embedding['learning']

        np.testing.assert_allclose(machine_vector[0:3], [-0.12538, 0.38888, 0.48011], rtol=1e-3)
        np.testing.assert_allclose(learning_vector[0:3], [0.047511, 0.1404, -0.11736], rtol=1e-3)


@pytest.mark.light
def test_load_word2vec():
    word2vec_path = os.path.expanduser('~/data/word2vec/GoogleNews-vectors-negative300.bin.gz')
    if os.path.isfile(word2vec_path):
        word_set = {'machine', 'learning'}
        word_to_embedding = load_word2vec(path=word2vec_path, words=word_set)

        machine_vector = word_to_embedding['machine']
        learning_vector = word_to_embedding['learning']

        np.testing.assert_allclose(machine_vector[0:3], [0.255859375, -0.0220947265625, 0.029052734375])
        np.testing.assert_allclose(learning_vector[0:3], [-0.08837890625, 0.1484375, -0.06298828125])


@pytest.mark.light
def test_load_glove():
    glove_path = 'data/glove/glove.6B.50d.txt.gz'
    if os.path.isfile(glove_path):
        word_set = {'house'}
        model = load_glove(path=glove_path, words=word_set)
        assert 'house' in model
        assert 0.60136 < model['house'][0] < 0.60138


if __name__ == '__main__':
    pytest.main([__file__])
