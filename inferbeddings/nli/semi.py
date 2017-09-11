# -*- coding: utf-8 -*-

import numpy as np

import logging

logger = logging.getLogger(__name__)





if __name__ == '__main__':
    for _ in range(8192):
        sizes1 = np.random.randint(0, 64, 1024)
        sizes2 = np.random.randint(0, 64, 1024)

        order = semi_sort(sizes1, sizes2)
        assert order.shape == (1024,)
        assert np.array_equal(np.sort(order), np.arange(1024))