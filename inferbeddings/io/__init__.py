# -*- coding: utf-8 -*-

from inferbeddings.io.base import iopen, read_triples, save
from inferbeddings.io.embeddings import load_glove, load_word2vec, load_glove_words, load_word2vec_words

__all__ = ['iopen',
           'read_triples',
           'save',
           'load_glove',
           'load_word2vec',
           'load_glove_words',
           'load_word2vec_words']
