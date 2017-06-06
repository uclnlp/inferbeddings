#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse
import requests
import operator

from tqdm import tqdm

from inferbeddings.nli.util import SNLI
import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('NLI Service', formatter_class=formatter)

    argparser.add_argument('--path', '-p', action='store', type=str, default='data/snli/snli_1.0_dev.jsonl.gz')

    args = argparser.parse_args(argv)

    path = args.path

    logger.debug('Reading corpus ..')
    instances, _, _ = SNLI.generate(train_path=path, valid_path=None, test_path=None)

    session = requests.Session()

    total, matches = 0, 0

    pbar = tqdm(instances)
    for instance in pbar:
        question, support, answer = instance['question'], instance['support'], instance['answer']
        res = session.post('http://127.0.0.1:8889/v1/nli', data={'sentence1': question, 'sentence2': support})
        prediction = sorted(res.json().items(), key=operator.itemgetter(1), reverse=True)[0][0]

        total += 1
        matches += 1 if answer == prediction else 0

        pbar.set_description('Accuracy: {:.2f}'.format(matches / total))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
