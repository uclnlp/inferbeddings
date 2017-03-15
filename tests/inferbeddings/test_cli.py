# -*- coding: utf-8 -*-

import pytest
import subprocess

import sys
sys.setrecursionlimit(65535)


def test_nations_cli():
    # Checking if results are still nice
    cmd = ['./bin/adv-cli.py',
           '--train', 'data/nations/stratified_folds/0/nations_train.tsv.gz',
           '--valid', 'data/nations/stratified_folds/0/nations_valid.tsv.gz',
           '--test', 'data/nations/stratified_folds/0/nations_test.tsv.gz',
           '--lr', '0.1',
           '--model', 'ComplEx',
           '--similarity', 'dot',
           '--margin', '1',
           '--embedding-size', '50',
           '--nb-epochs', '10']  # 1000
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    # Hits@10 should be at least 90% even after a limited number of epochs
    assert float(err.split()[-1][:-1]) > 90.0

    cmd = ['./bin/adv-cli.py',
           '--train', 'data/nations/stratified_folds/0/nations_train.tsv.gz',
           '--valid', 'data/nations/stratified_folds/0/nations_valid.tsv.gz',
           '--test', 'data/nations/stratified_folds/0/nations_test.tsv.gz',
           '--lr', '0.1',
           '--model', 'TransE',
           '--similarity', 'l1',
           '--margin', '1',
           '--embedding-size', '20',
           '--nb-epochs', '50']  # 1000
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    # Hits@10 should be at least 90% even after a limited number of epochs
    assert float(err.split()[-1][:-1]) > 90.0

    cmd = ['./bin/adv-cli.py',
           '--train', 'data/nations/stratified_folds/0/nations_train.tsv.gz',
           '--valid', 'data/nations/stratified_folds/0/nations_valid.tsv.gz',
           '--test', 'data/nations/stratified_folds/0/nations_test.tsv.gz',
           '--lr', '0.1',
           '--model', 'DistMult',
           '--similarity', 'dot',
           '--margin', '1',
           '--embedding-size', '50',
           '--nb-epochs', '50']  # 1000
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    # Hits@10 should be at least 85% even after a limited number of epochs
    assert float(err.split()[-1][:-1]) > 85.0

if __name__ == '__main__':
    pytest.main([__file__])
