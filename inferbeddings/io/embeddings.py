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
    logger.info('Loading GloVe embeddings ..')
    model = {}
    for n, line in enumerate(stream):
        split_line = line.decode('utf-8').split()
        word = split_line[0]
        if words is None or word in words:
            embedding = [float(val) for val in split_line[1:]]
            model[word] = embedding
        if n % 1000 == 0:
            logger.info('  {}k vectors processed...\r'.format(str(n // 1000)))
    logger.info('Done loading GloVe embeddings.')
    return model
