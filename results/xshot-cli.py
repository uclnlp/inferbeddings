#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import subprocess

import logging


def main(argv):
    p = subprocess.Popen(['sh', '-c', './tools/parse_results_filtered.sh logs/ucl_fb15k_adv_v1/*.log'],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
