# -*- coding: utf-8 -*-

import inferbeddings.nli.util as util

import logging

import pytest

logger = logging.getLogger(__name__)


def test_snli_lower():
    train, dev, test = util.SNLI.generate(is_lower=True)
    all_instances = train + dev + test

    token_set = set()
    for instance in all_instances:
        token_set |= set(instance['sentence1_parse_tokens'])
        token_set |= set(instance['sentence2_parse_tokens'])

    print(len(token_set))
    assert len(token_set) == 36988


def test_snli_tiny_lower():
    path = 'data/snli/tiny/tiny.jsonl.gz'
    train, dev, test = util.SNLI.generate(train_path=path, valid_path=path, test_path=path, is_lower=True)
    all_instances = train + dev + test

    token_set = set()
    for instance in all_instances:
        token_set |= set(instance['sentence1_parse_tokens'])
        token_set |= set(instance['sentence2_parse_tokens'])

    print(len(token_set))
    assert len(token_set) == 392

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    pytest.main([__file__])
