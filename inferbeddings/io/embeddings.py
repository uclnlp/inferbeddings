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
    word_to_embedding = {}

    for n, line in enumerate(stream):
        if not isinstance(line, str):
            line = line.decode('utf-8')
        split_line = line.split(' ')
        word = split_line[0]

        if words is None or word in words:
            try:
                word_to_embedding[word] = [float(f) for f in split_line[1:]]
            except ValueError:
                logger.error('{}\t{}\t{}'.format(n, word, str(split_line)))

    return word_to_embedding


def load_word2vec(stream, words=None):
    """
    Loads word2vec word embeddings.

    :param stream: An opened stream to the GloVe file.
    :param words: Words in the existing vocabulary.
    :return: dict {word: embedding}
    """
    word_to_embedding = {}

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

        # if not isinstance(word, str):
            # word = word.decode('utf-8')

        if words is None or word in words:
            try:
                vector = np.fromstring(stream.read(byte_size), dtype=np.float32)
                word_to_embedding[word] = vector.tolist()
            except ValueError:
                logger.error('{}\t{}\t{}'.format(n, word, str(byte_size)))

    return word_to_embedding
