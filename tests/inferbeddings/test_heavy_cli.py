# -*- coding: utf-8 -*-

import pytest
import subprocess

import sys
sys.setrecursionlimit(65535)


def test_wordnet_complex_cli():
    # Checking if results are still nice
    cmd = ['./bin/adv-cli.py',
           '--train', 'data/wn18/wordnet-mlj12-train.txt',
           '--test', 'data/wn18/wordnet-mlj12-test.txt',
           '--lr', '0.1',
           '--model', 'ComplEx',
           '--similarity', 'dot',
           '--margin', '5',
           '--embedding-size', '100',
           '--nb-epochs', '100']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    # Hits@10 should be at least 90% even after a limited number of epochs
    assert float(err.split()[-1][:-1]) > 91.0


def test_wordnet_translating_cli():
    # Checking if results are still nice
    cmd = ['./bin/adv-cli.py',
           '--train', 'data/wn18/wordnet-mlj12-train.txt',
           '--test', 'data/wn18/wordnet-mlj12-test.txt',
           '--lr', '0.1',
           '--model', 'TransE',
           '--similarity', 'l1',
           '--margin', '2',
           '--embedding-size', '50',
           '--nb-epochs', '1000']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    # Hits@10 should be at least 90% even after a limited number of epochs
    assert float(err.split()[-1][:-1]) > 93.0


if __name__ == '__main__':
    pytest.main([__file__])
