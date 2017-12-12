# -*- coding: utf-8 -*

import numpy as np

import logging

logger = logging.getLogger(__name__)


def decode(sequence_embedding, embedding_matrix, index_to_token=None):
    sequence_length = sequence_embedding.shape[0]
    embedding_size = sequence_embedding.shape[1]
    vocab_size = embedding_matrix.shape[0]

    assert embedding_size == embedding_matrix.shape[1]
    assert vocab_size > 0

    res = []
    for i in range(sequence_length):
        n_idx = find_nearest(sequence_embedding[i, :], embedding_matrix)
        res += [index_to_token[n_idx]] if index_to_token else [n_idx]

    return res


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
