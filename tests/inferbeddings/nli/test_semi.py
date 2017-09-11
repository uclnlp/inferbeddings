# -*- coding: utf-8 -*-

import numpy as np
import inferbeddings.nli.util as util

import logging

import pytest

logger = logging.getLogger(__name__)


def test_semi_sort():
    for _ in range(8192):
        sizes1 = np.random.randint(0, 64, 1024)
        sizes2 = np.random.randint(0, 64, 1024)

        order = util.semi_sort(sizes1, sizes2)
        assert order.shape == (1024,)
        assert np.array_equal(np.sort(order), np.arange(1024))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    pytest.main([__file__])
