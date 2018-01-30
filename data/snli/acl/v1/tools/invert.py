#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse
import json

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    def fmt(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('SNLI Inverter', formatter_class=fmt)

    argparser.add_argument('--path', '-p', action='store', type=str, default='/dev/stdin')

    args = argparser.parse_args(argv)

    path = args.path

    obj_lst = []
    with open(path, 'rb') as f:
        for line in f:
            dl = line.decode('utf-8')
            obj = json.loads(dl)
            obj_lst += [obj]

    def invert(_obj):
        import copy
        i_obj = copy.deepcopy(_obj)
        for a, b in [(1, 2), (2, 1)]:
            for suf in ['', '_binary_parse', '_parse']:
                i_obj['sentence{}{}'.format(a, suf)] = _obj['sentence{}{}'.format(b, suf)]
        return i_obj

    i_obj_lst = [invert(obj) for obj in obj_lst]

    for obj in i_obj_lst:
        s_obj = json.dumps(obj)
        print(s_obj, end='')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
