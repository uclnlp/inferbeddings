# -*- coding: utf-8 -*-

import pytest
from inferbeddings.io import iopen, load_glove


def test_load_glove():
    with iopen('data/glove/glove.6B.50d.txt.gz', 'r') as f:
        model = load_glove(f, {'house'})

    assert 'house' in model
    assert 0.60136 < model['house'][0] < 0.60138

if __name__ == '__main__':
    pytest.main([__file__])
