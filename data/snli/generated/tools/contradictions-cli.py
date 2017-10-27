#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import os
import sys

import gzip
import json

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    def fmt(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Tool for generating new SNLI examples', formatter_class=fmt)
    argparser.add_argument('path', action='store', type=str, default=None)
    argparser.add_argument('--invert', '-i', action='store_true')
    argparser.add_argument('--retain', '-r', action='store_true')

    args = argparser.parse_args(argv)

    path = args.path
    is_invert = args.invert
    is_retain = args.retain

    instances = []

    with gzip.open(path, 'rb') as f:
        for line in f:
            decoded_line = line.decode('utf-8')
            obj = json.loads(decoded_line)
            instances += [obj]

    for obj in instances:
        if obj['gold_label'] in {'contradiction'}:
            new_obj = obj.copy()

            if is_invert:
                sentence1 = obj['sentence1']
                sentence1_parse = obj['sentence1_parse']
                sentence1_binary_parse = obj['sentence1_binary_parse']

                sentence2 = obj['sentence2']
                sentence2_parse = obj['sentence2_parse']
                sentence2_binary_parse = obj['sentence2_binary_parse']

                new_obj.update({
                    'sentence1': sentence2,
                    'sentence1_parse': sentence2_parse,
                    'sentence1_binary_parse': sentence2_binary_parse,
                    'sentence2': sentence1,
                    'sentence2_parse': sentence1_parse,
                    'sentence2_binary_parse': sentence1_binary_parse
                })

            print(json.dumps(new_obj))
        else:
            if is_retain:
                print(json.dumps(obj))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
