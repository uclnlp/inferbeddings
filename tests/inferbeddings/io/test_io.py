# -*- coding: utf-8 -*-

import pytest
from inferbeddings.io import iopen, load_glove, load_word2vec


def test_load_glove():
    with iopen('data/glove/glove.6B.50d.txt.gz', 'r') as f:
        model = load_glove(f, {'house'})

    assert 'house' in model
    assert 0.60136 < model['house'][0] < 0.60138


# def test_load_word2vec():
#     with iopen('data/word2vec/GoogleNews-vectors-negative300.bin.gz') as f:
#         model = load_word2vec(f, {'house'})
#     assert 'house' in model
#     print(model['house'])

if __name__ == '__main__':
    pytest.main([__file__])
