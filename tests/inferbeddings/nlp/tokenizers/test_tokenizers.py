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


def test_oov_tokenizer_1():
    vocabulary = {'hello'}

    tokenizer = OOVTokenizer(has_bos=True, has_eos=True, has_unk=True, vocabulary=vocabulary)
    tokenizer.fit_on_texts(['Hello world!'])

    assert tokenizer.texts_to_sequences(['Hello world sup?']) == [[1, 5, 4, 3, 2]]
    assert tokenizer.word_index == {'hello': 5, 'world': 4}

    assert tokenizer.nb_iv_words == 1  # 'hello'
    assert tokenizer.nb_oov_words == 5  # 0 (PAD), BOS, EOS, UNK, 'world'


def test_oov_tokenizer_2():
    vocabulary = {'hello'}

    tokenizer = OOVTokenizer(has_bos=False, has_eos=True, has_unk=True, vocabulary=vocabulary)
    tokenizer.fit_on_texts(['Hello world!'])

    assert tokenizer.texts_to_sequences(['Hello world sup?']) == [[4, 3, 2, 1]]
    assert tokenizer.word_index == {'hello': 4, 'world': 3}

    assert tokenizer.nb_iv_words == 1  # 'hello'
    assert tokenizer.nb_oov_words == 4  # 0 (PAD), EOS, UNK, 'world'


def test_oov_tokenizer_3():
    vocabulary = {'hello'}

    tokenizer = OOVTokenizer(has_bos=False, has_eos=False, has_unk=True, vocabulary=vocabulary)
    tokenizer.fit_on_texts(['Hello world!'])

    assert tokenizer.texts_to_sequences(['Hello world sup?']) == [[3, 2, 1]]
    assert tokenizer.word_index == {'hello': 3, 'world': 2}

    assert tokenizer.nb_iv_words == 1  # 'hello'
    assert tokenizer.nb_oov_words == 3  # 0 (PAD), UNK, 'world'


def test_oov_tokenizer_4():
    vocabulary = {'hello'}

    tokenizer = OOVTokenizer(has_bos=False, has_eos=False, has_unk=False, vocabulary=vocabulary)
    tokenizer.fit_on_texts(['Hello world!'])

    assert tokenizer.texts_to_sequences(['Hello world sup?']) == [[2, 1]]
    assert tokenizer.word_index == {'hello': 2, 'world': 1}

    assert tokenizer.nb_iv_words == 1  # 'hello'
    assert tokenizer.nb_oov_words == 2  # 0 (PAD), 'world'


def test_oov_tokenizer_5():
    vocabulary = {'hello', 'world'}

    tokenizer = OOVTokenizer(has_bos=False, has_eos=False, has_unk=False, vocabulary=vocabulary)
    tokenizer.fit_on_texts(['Hello world!'])

    assert tokenizer.texts_to_sequences(['Hello world sup?']) == [[1, 2]]
    assert tokenizer.word_index == {'hello': 1, 'world': 2}

    assert tokenizer.nb_iv_words == 2  # 'hello', 'world'
    assert tokenizer.nb_oov_words == 1  # 0 (PAD)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    pytest.main([__file__])
