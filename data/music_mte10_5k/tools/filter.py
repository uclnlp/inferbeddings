#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

import sys
import gzip
import bz2

import codecs
import argparse
import logging


def iopen(file, *args, **kwargs):
    _open = open
    if file.endswith('.gz'):
        _open = gzip.open
    elif file.endswith('.bz2'):
        _open = bz2.open
    return _open(file, *args, **kwargs)


def read_triples(path):
    logging.info('Acquiring %s ..' % path)
    with iopen(path, mode='r') as f:
        lines = f.readlines()
    triples = [(s.strip(), p.strip(), o.strip()) for [s, p, o] in [l.split('\t') for l in lines]]
    return triples


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Filtering tool for filtering away infrequent entities from Knowledge Graphs',
                                        formatter_class=formatter)
    argparser.add_argument('triples', type=str, help='File containing triples')
    argparser.add_argument('entities', type=str, help='File containing entities')

    args = argparser.parse_args(argv)

    triples_path = args.triples
    entities_path = args.entities

    triples = read_triples(triples_path)

    with iopen(entities_path, mode='r') as f:
        entities = set([line.strip() for line in f.readlines()])

    for (s, p, o) in triples:
        if s in entities and o in entities:
            print("%s\t%s\t%s" % (s, p, o))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
