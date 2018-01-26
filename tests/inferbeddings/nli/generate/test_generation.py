# -*- coding: utf-8 -*-

import pytest

import sys
from urllib3.exceptions import NewConnectionError, MaxRetryError

from nltk.parse.corenlp import CoreNLPParser
from inferbeddings.nli.generate import operators

import logging


def test_generate():
    try:
        parser = CoreNLPParser(url='http://hamburg.vpn:9000')
        parser.raw_parse('This is a test sentence.')
    except Exception:
        # We were not able to connect, so we set parser to None
        parser = None

    if parser is not None:
        tree1, = parser.raw_parse('Hi, my name is Squippy!')
        tree2, = parser.raw_parse('No, I do not believe that your name is Squippy!')

        tree_lst = operators.combine_trees(tree1, tree2)
        for tree in set([str(t.leaves()) for t in tree_lst]):
            # print(tree)
            pass

        tree_lst = operators.remove_subtree(tree1)
        for tree in set([str(t.leaves()) for t in tree_lst]):
            print(tree)

if __name__ == '__main__':
    # pytest.main([__file__])
    test_generate()
