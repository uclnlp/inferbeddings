# -*- coding: utf-8 -*-

import pytest
import subprocess

import sys
sys.setrecursionlimit(65535)


@pytest.mark.light
def test_nli_cli():
    # Checking if results are still nice
    cmd = ['./bin/nli-cli.py',
           '-f',
           '-n',
           '-m', 'ff-damp',
           '--batch-size', '32',
           '--representation-size', '200',
           '--optimizer', 'adagrad',
           '--learning-rate', '0.05',
           '--restore', './models/nli/damp_v1.ckpt',
           '--use-masking',
           '--nb-epochs', '0']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    print(err)

    print(err.split()[-1][:-1])

if __name__ == '__main__':
    pytest.main([__file__])
