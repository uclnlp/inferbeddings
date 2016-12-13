#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import logging

from inferbeddings import parse


def main(argv):
    clause_str = '/p.q(x, y) :- /p.q(x, z), q(z, a), r(a, y)'
    clause = parse.parse_clause(clause_str)
    print(type(clause))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
