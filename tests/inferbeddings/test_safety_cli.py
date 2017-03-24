# -*- coding: utf-8 -*-

import pytest
import subprocess

import sys
sys.setrecursionlimit(65535)


def test_wn18_cli():
    # Checking if results are still nice
    cmd = ['./bin/adv-cli.py',
           '--train', 'data/wn18/wordnet-mlj12-train.txt',
           '--lr', '0.1',
           '--model', 'TransE',
           '--similarity', 'l1',
           '--margin', '2',
           '--embedding-size', '50',
           '--nb-epochs', '10']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    for line in str(err).split("\\n"):
        if "Epoch: 1/1\\tLoss:" in line:
            assert line.split()[2] == "3.3778"
            assert line.split()[4] == "0.5889"
        if "Epoch: 2/1\\tLoss:" in line:
            assert line.split()[2] == "1.3837"
            assert line.split()[4] == "0.1561"
        if "Epoch: 3/1\\tLoss:" in line:
            assert line.split()[2] == "0.5752"
            assert line.split()[4] == "0.0353"
        if "Epoch: 4/1\\tLoss:" in line:
            assert line.split()[2] == "0.2984"
            assert line.split()[4] == "0.0071"
        if "Epoch: 5/1\\tLoss:" in line:
            assert line.split()[2] == "0.1842"
        if "Epoch: 6/1\\tLoss:" in line:
            assert line.split()[2] == "0.1287"
        if "Epoch: 7/1\\tLoss:" in line:
            assert line.split()[2] == "0.0980"
        if "Epoch: 8/1\\tLoss:" in line:
            assert line.split()[2] == "0.0795"
        if "Epoch: 9/1\\tLoss:" in line:
            assert line.split()[2] == "0.0653"
        if "Epoch: 10/1\\tLoss:" in line:
            assert line.split()[2] == "0.0562"

if __name__ == '__main__':
    pytest.main([__file__])
