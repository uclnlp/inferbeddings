# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from inferbeddings.nli import tfutil

import pytest

import logging


@pytest.mark.light
def test_nli_alpha_beta():
    rs = np.random.RandomState(0)
    weight_matrix = rs.rand(1, 5, 4)

    weight_matrix_1 = np.transpose(
        np.exp(weight_matrix - weight_matrix.max(1)[:, None, :]),
        (1, 2, 0))
    weight_matrix_2 = np.transpose(
        np.exp(weight_matrix - weight_matrix.max(2)[:, :, None]),
        (1, 2, 0))

    assert weight_matrix_1.shape == (5, 4, 1)
    assert weight_matrix_2.shape == (5, 4, 1)

    alpha = weight_matrix_1 / (weight_matrix_1.sum(0)[None, :, :])
    beta = weight_matrix_2 / (weight_matrix_2.sum(1)[:, None, :])

    assert alpha.shape == (5, 4, 1)
    assert beta.shape == (5, 4, 1)

    print(alpha[:, :, 0])
    print(beta[:, :, 0])

    with tf.Session() as session:
        x = session.run(tfutil.attention_softmax3d(weight_matrix))
        np.testing.assert_allclose(x[0, :, :], beta[:, :, 0])
        x = session.run(tfutil.attention_softmax3d(np.transpose(weight_matrix, (0, 2, 1))))
        np.testing.assert_allclose(np.transpose(x[0, :, :], (1, 0)), alpha[:, :, 0])

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    pytest.main([__file__])
