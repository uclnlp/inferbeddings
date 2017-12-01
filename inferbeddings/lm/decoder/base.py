# -*- coding: utf-8 -*

import numpy as np

import logging

logger = logging.getLogger(__name__)


def find_nearest(vector, embedding_matrix):
    """
    Args:
        vector: vector with shape [embedding_size]
        embedding_matrix: matrix with shape [vocab_size, embedding_size]

    Returns:
        index of the row in embedding_matrix which is closest to vector
    """
    assert vector.shape[0] == embedding_matrix.shape[1]
    idx = np.abs(embedding_matrix - vector).sum(1).argmin()
    return idx
