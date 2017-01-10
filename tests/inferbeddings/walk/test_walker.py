# -*- coding: utf-8 -*-

import pytest

from inferbeddings.walk import BidirectionalWalker


def test_walker():
    triples = [
        ('A', 'p', 'B'),
        ('B', 'p', 'C')
    ]

    walkgen = BidirectionalWalker(triples=triples)
    steps, [source, target] = walkgen(length=1)

    predicate, is_inverse = steps

    if is_inverse is True:
        assert((target, predicate, source) in triples)
    else:
        assert ((source, predicate, target) in triples)


if __name__ == '__main__':
    pytest.main([__file__])
