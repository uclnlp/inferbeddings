#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import os
import sys

import requests

import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(os.path.basename(sys.argv[0]))


def call_service(url, sentence1, sentence2):
    data = {
        'sentence1': sentence1,
        'sentence2': sentence2
    }
    res = requests.post(url, data=data)
    res_json = res.json()
    return res_json


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('NLI Client', formatter_class=formatter)

    argparser.add_argument('s1', action='store', type=str)
    argparser.add_argument('s2', action='store', type=str)
    argparser.add_argument('--url', '-u', action='store', type=str, default='http://127.0.0.1:8889/v1/nli')

    args = argparser.parse_args(argv)

    sentence1 = args.s1
    sentence2 = args.s2

    url = args.url

    prediction = call_service(url=url, sentence1=sentence1, sentence2=sentence2)
    print(prediction)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
