#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from nltk.parse.corenlp import CoreNLPParser
import logging


def insert(tree1, st_idx, tree2):
    len_st = len(list(tree1.subtrees())[st_idx])
    res = []
    for i in range(len_st + 1):
        tree1_cp, tree2_cp = tree1.copy(deep=True), tree2.copy(deep=True)
        st = list(tree1_cp.subtrees())[st_idx]
        st.insert(i, tree2_cp)
        res += [tree1_cp]
    return res

def combine


def main(argv):
    parser = CoreNLPParser(url='http://hamburg.vpn:9000')

    parse1, = parser.raw_parse('Hi, my name is Squippy!')
    parse2, = parser.raw_parse('No, I do not believe that your name is Squippy, liar!')
    parse3, = parser.raw_parse('XXX')

    parse_lst = insert(parse1, 1, parse3)
    for parse in parse_lst:
        parse.pretty_print()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
