# -*- coding: utf-8 -*-

import numpy as np


def make_batches(size, batch_size):
    """
    Returns a list of batch indices (tuples of indices).

    :param size: Size of the dataset (number of examples).
    :param batch_size: Batch size.
    :return: List of batch indices (tuples of indices).
    """
    nb_batch = int(np.ceil(size / float(batch_size)))
    res = [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]
    return res
