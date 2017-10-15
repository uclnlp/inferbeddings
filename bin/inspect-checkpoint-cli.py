#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import os
import sys

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Checkpoint Inspector', formatter_class=formatter)
    argparser.add_argument('file_name', action='store', type=str)
    args = argparser.parse_args(argv)

    file_name = args.file_name

    print_tensors_in_checkpoint_file(file_name=file_name, tensor_name='', all_tensors=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
