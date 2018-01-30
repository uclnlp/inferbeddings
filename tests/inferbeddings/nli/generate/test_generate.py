# -*- coding: utf-8 -*-

import pytest

from inferbeddings.nli.generate import operators
from inferbeddings.nli.generate.parser import Parser


def test_generate():
    parser = Parser()

    tree1 = parser.parse('Hi, my name is Squippy!')
    tree2 = parser.parse('No, I do not believe that your name is Squippy!')

    tree_lst = operators.combine_trees(tree1, tree2)
    for tree in tree_lst:
        assert 'Hi' in tree.leaves()

    tree_lst = operators.combine_trees(tree2, tree1)
    for tree in tree_lst:
        assert 'No' in tree.leaves()

    tree_lst = operators.remove_subtree(tree1)
    for tree in tree_lst:
        assert len(tree.leaves()) <= 6

if __name__ == '__main__':
    pytest.main([__file__])
