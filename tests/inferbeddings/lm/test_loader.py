# -*- coding: utf-8 -*-

import numpy as np

import pytest

from inferbeddings.lm.loader2 import SNLILoader


def test_snli_loader():
    import pickle

    path = 'models/snli/dam_1/dam_1_index_to_token.p'
    with open(path, 'rb') as f:
        index_to_token = pickle.load(f)

    token_to_index = {token: index for index, token in index_to_token.items()}

    loader = SNLILoader(path='data/snli/snli_1.0_test.jsonl.gz', token_to_index=token_to_index)

    loader.create_batches()

    for batch_id in range(loader.pointer, loader.num_batches):
        x, y = loader.next_batch()
        assert x.shape == y.shape
        for j in range(x.shape[0]):
            assert x[j].shape == y[j].shape


if __name__ == '__main__':
    pytest.main([__file__])
