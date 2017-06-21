# -*- coding: utf-8 -*-

from inferbeddings.nlp import Tokenizer, OOVTokenizer

import logging

import pytest

logger = logging.getLogger(__name__)


def test_tokenizer():
    tokenizer = Tokenizer(has_bos=True, has_eos=True, has_unk=True)
    tokenizer.fit_on_texts(['Hello world!'])

    assert tokenizer.texts_to_sequences(['Hello world sup?']) == [[1, 4, 5, 3, 2]]
    assert tokenizer.word_index == {'hello': 4, 'world': 5}


def test_oov_tokenizer():
    vocabulary = {'hello'}

    tokenizer = OOVTokenizer(has_bos=True, has_eos=True, has_unk=True, vocabulary=vocabulary)
    tokenizer.fit_on_texts(['Hello world!'])

    assert tokenizer.texts_to_sequences(['Hello world sup?']) == [[1, 5, 4, 3, 2]]
    assert tokenizer.word_index == {'hello': 5, 'world': 4}


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    pytest.main([__file__])
