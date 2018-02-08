# -*- coding: utf-8 -*-

import pytest

import pickle
import nltk
import numpy as np

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
    g = Generator(token_to_index=token_to_index, nb_corruptions=0)

    rs = np.random.RandomState(0)

    for nb_corruptions in [8, 16, 32, 64, 128, 256, 512]:
        g.nb_corruptions = nb_corruptions

        sentence1 = 'The girl runs happily on the grass close to a white box .'
        sentence2 = 'The girl is happy .'

        is_idxs = rs.randint(0, 2)
        if is_idxs:
            sentence1 = [token_to_index[tkn] for tkn in sentence1.split()]
            sentence2 = [token_to_index[tkn] for tkn in sentence2.split()]

        corr1, corr2 = g.flip(sentence1=sentence1, sentence2=sentence2)

        assert len(corr1) == nb_corruptions
        assert len(corr2) == nb_corruptions

        tokenizer = nltk.tokenize.TreebankWordTokenizer()

        for e in corr1:
            e_tkns = e if is_idxs else tokenizer.tokenize(e)
            s_tkns = sentence1 if is_idxs else tokenizer.tokenize(sentence1)
            assert len(e_tkns) == len(s_tkns)

        for e in corr2:
            e_tkns = e if is_idxs else tokenizer.tokenize(e)
            s_tkns = sentence2 if is_idxs else tokenizer.tokenize(sentence2)
            assert len(e_tkns) == len(s_tkns)

        corr1, corr2 = g.remove(sentence1=sentence1, sentence2=sentence2)

        assert len(corr1) == nb_corruptions
        assert len(corr2) == nb_corruptions

        for e in corr1:
            e_tkns = e if is_idxs else tokenizer.tokenize(e)
            s_tkns = sentence1 if is_idxs else tokenizer.tokenize(sentence1)
            assert len(e_tkns) <= len(s_tkns)

        for e in corr2:
            e_tkns = e if is_idxs else tokenizer.tokenize(e)
            s_tkns = sentence2 if is_idxs else tokenizer.tokenize(sentence2)
            assert len(e_tkns) <= len(s_tkns)

        corr1, corr2 = g.combine(sentence1=sentence1, sentence2=sentence2)

        assert len(corr1) == nb_corruptions
        assert len(corr2) == nb_corruptions

        for e in corr1:
            e_tkns = e if is_idxs else tokenizer.tokenize(e)
            s_tkns = sentence1 if is_idxs else tokenizer.tokenize(sentence1)
            assert len(e_tkns) >= len(s_tkns)

        for e in corr2:
            e_tkns = e if is_idxs else tokenizer.tokenize(e)
            s_tkns = sentence2 if is_idxs else tokenizer.tokenize(sentence2)
            assert len(e_tkns) >= len(s_tkns)

if __name__ == '__main__':
    # pytest.main([__file__])
    test_generator()
