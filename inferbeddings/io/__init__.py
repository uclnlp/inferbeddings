# -*- coding: utf-8 -*-

from inferbeddings.io.base import iopen, read_triples, save
from inferbeddings.io.embeddings import load_glove, load_word2vec

__all__ = ['iopen',
           'read_triples',
           'save',
           'load_glove',
           'load_word2vec']
