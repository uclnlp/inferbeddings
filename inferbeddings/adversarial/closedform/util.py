# -*- coding: utf-8 -*-

import numpy as np


def dot3(a, b, c):
    return np.sum(a * b * c)


def score_complex(subject_embedding, predicate_embedding, object_embedding):
    n = subject_embedding.shape[0]
    s_re, s_im = subject_embedding[:n // 2], subject_embedding[n // 2:]
    p_re, p_im = predicate_embedding[:n // 2], predicate_embedding[n // 2:]
    o_re, o_im = object_embedding[:n // 2], object_embedding[n // 2:]
    return dot3(s_re, p_re, o_re) + dot3(s_re, p_im, o_im) + dot3(s_im, p_re, o_im) - dot3(s_im, p_im, o_re)