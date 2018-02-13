#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob

import logging


def main(argv):
    paths = list(glob.iglob('out_nli/**/*.log', recursive=True))

    for path in paths:
        base = os.path.basename(path)
        noext = os.path.splitext(base)[0]
        print(path, base, noext)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
