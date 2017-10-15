#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import os
import sys

import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    pass

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
