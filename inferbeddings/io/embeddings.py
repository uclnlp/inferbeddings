# -*- coding: utf-8 -*-

import numpy as np

import logging

logger = logging.getLogger(__name__)


def load_glove(stream, words=None):
    """
    Loads GloVe word embeddings.

    :param stream: An opened stream to the GloVe file.
    :param words: Words in the existing vocabulary.
    :return: dict {word: embedding}
    """
    word2embedding = {}
    for n, line in enumerate(stream):
        split_line = line.decode('utf-8').split()
        word = split_line[0]
        if words is None or word in words:
            word2embedding[word] = [float(val) for val in split_line[1:]]
    return word2embedding


def load_word2vec(stream, words=None):
    """
    Loads word2vec word embeddings.

    :param stream: An opened stream to the GloVe file.
    :param words: Words in the existing vocabulary.
    :return: dict {word: embedding}
    """
    word2embedding = {}

    vec_n, vec_size = map(int, stream.readline().split())
    byte_size = vec_size * 4

    for n in range(vec_n):
        word = b''

        while True:
            c = stream.read(1)
            if c == b' ':
                break
            else:
                word += c
        word = word.decode('utf-8')

        vector = np.fromstring(stream.read(byte_size), dtype=np.float32)
        if words is None or word in words:
            word2embedding[word] = vector.tolist()
    return word2embedding
