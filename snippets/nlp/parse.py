#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from nltk.parse.corenlp import CoreNLPParser

import logging


def _insert(tree1, st_idx, tree2):
    len_st = len(list(tree1.subtrees())[st_idx])
    res = []
    for i in range(len_st + 1):
        tree1_cp, tree2_cp = tree1.copy(deep=True), tree2.copy(deep=True)
        st = list(tree1_cp.subtrees())[st_idx]
        st.insert(i, tree2_cp)
        res += [tree1_cp]
    return res


def combine(tree1, tree2):
    nb_sts1 = len(list(tree1.subtrees()))
    nb_sts2 = len(list(tree2.subtrees()))

    res = []
    for i in range(nb_sts1):
        res += _insert(tree1, i, tree2)
    for i in range(nb_sts2):
        res += _insert(tree2, i, tree1)

    _res = set(tuple(t.leaves()) for t in res)
    return [list(r) for r in sorted(_res)]


def main(argv):
    parser = CoreNLPParser(url='http://hamburg.vpn:9000')

    parse1, = parser.raw_parse('Hi, my name is Squippy!')
    parse2, = parser.raw_parse('No, I do not believe that your name is Squippy, liar!')
    parse3, = parser.raw_parse('XXX')

    parse_lst = combine(parse1, parse2)
    for p in parse_lst:
        print(p)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
