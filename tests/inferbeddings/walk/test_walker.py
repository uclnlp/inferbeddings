# -*- coding: utf-8 -*-

import pytest

from inferbeddings.walk import BidirectionalWalker


def test_walker():
    triples = [
        ('A', 'p', 'B'),
        ('B', 'p', 'C')
    ]

    walkgen = BidirectionalWalker(triples=triples)
    walk = walkgen(length=1)

    assert len(walk) == 2


if __name__ == '__main__':
    pytest.main([__file__])
