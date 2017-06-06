#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import os
import sys

from inferbeddings.nli.util import SNLI
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('NLI Service', formatter_class=formatter)

    argparser.add_argument('--train', '-t', action='store', type=str, default='data/snli/snli_1.0_train.jsonl.gz')
    argparser.add_argument('--valid', '-v', action='store', type=str, default='data/snli/snli_1.0_dev.jsonl.gz')
    argparser.add_argument('--test', '-T', action='store', type=str, default='data/snli/snli_1.0_test.jsonl.gz')

    args = argparser.parse_args(argv)

    train_path, valid_path, test_path = args.train, args.valid, args.test

    logger.debug('Reading corpus ..')
    train_instances, dev_instances, test_instances =\
        SNLI.generate(train_path=train_path, valid_path=valid_path, test_path=test_path)

    print(train_instances[0])


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
