# -*- coding: utf-8 -*-

import pytest

import pickle
import nltk

from inferbeddings.nli.generate.generator import Generator


def test_generator():
    vocabulary_path = 'models/snli/dam_1/dam_1_index_to_token.p'

    with open(vocabulary_path, 'rb') as f:
        index_to_token = pickle.load(f)

    index_to_token.update({
        0: '<PAD>',
        1: '<BOS>',
        2: '<UNK>'
    })

    token_to_index = {v: k for k, v in index_to_token.items()}

    nb_corruptions = 32
    g = Generator(token_to_index=token_to_index, nb_corruptions=nb_corruptions)

    sentence1 = 'The girl runs happily on the grass close to a white box .'
    sentence2 = 'The girl is happy .'

    corr1, corr2 = g.flip(sentence1=sentence1, sentence2=sentence2)

    assert len(corr1) == nb_corruptions
    assert len(corr2) == nb_corruptions

    tokenizer = nltk.tokenize.TreebankWordTokenizer()

    for e in corr1:
        e_tkns = tokenizer.tokenize(e)
        s_tkns = tokenizer.tokenize(sentence1)
        assert len(e_tkns) == len(s_tkns)

    for e in corr2:
        e_tkns = tokenizer.tokenize(e)
        s_tkns = tokenizer.tokenize(sentence2)
        assert len(e_tkns) == len(s_tkns)

    corr1, corr2 = g.remove(sentence1=sentence1, sentence2=sentence2)

    assert len(corr1) == nb_corruptions
    assert len(corr2) == nb_corruptions

    for e in corr1:
        e_tkns = tokenizer.tokenize(e)
        s_tkns = tokenizer.tokenize(sentence1)
        assert len(e_tkns) <= len(s_tkns)

    for e in corr2:
        e_tkns = tokenizer.tokenize(e)
        s_tkns = tokenizer.tokenize(sentence2)
        assert len(e_tkns) <= len(s_tkns)


if __name__ == '__main__':
    # pytest.main([__file__])
    test_generator()
