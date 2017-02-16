# -*- coding: utf-8 -*-

import numpy as np

import logging

logger = logging.getLogger(__name__)


def load_glove(stream, vocab):
    """Loads GloVe file and merges it if optional vocabulary
    Args:
        stream (iterable): An opened filestream to the GloVe file.
        vocab (dict=None): Word2idx dict of existing vocabulary.
    Returns:
        return_vocab (Vocabulary), lookup (matrix); Vocabulary contains the
                     word2idx and the matrix contains the embedded words.
    """
    logger.info('[Loading GloVe]')
    word2idx = {}
    first_line = stream.readline()
    dim = len(first_line.split()) - 1
    lookup = np.empty([len(vocab) + 1, dim], dtype=np.float)
    lookup[0] = np.fromstring(first_line.split(maxsplit=1)[1], sep=' ')
    word2idx[first_line.split(maxsplit=1)[0]] = 0
    n = 1
    for line in stream:
        word, vec = line.rstrip().split(maxsplit=1)
        if vocab is None or word in vocab and word not in word2idx:
            # word = word.decode('utf-8')
            idx = len(word2idx)
            word2idx[word] = idx
            lookup[idx] = np.fromstring(vec, sep=' ')
        n += 1
        if n % 100000 == 0:
            logger.info('  ' + str(n // 1000) + 'k vectors processed...\r')
    np.resize(lookup, (len(word2idx), dim))
    # lookup.resize((len(word2idx), dim))
    return_vocab = word2idx
    logger.info('[Loading GloVe DONE]')
    return return_vocab, lookup
