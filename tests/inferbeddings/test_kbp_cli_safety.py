# -*- coding: utf-8 -*-

import pytest
import subprocess

import sys
sys.setrecursionlimit(65535)


@pytest.mark.light
def test_wn18_cli():
    # Checking if results are still the same
    cmd = ['./bin/kbp-cli.py',
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
            assert line.split()[4] == "0.0069"
        if "Epoch: 5/1\\tLoss:" in line:
            assert line.split()[2] == "0.1840"
        if "Epoch: 6/1\\tLoss:" in line:
            assert line.split()[2] == "0.1286"
        if "Epoch: 7/1\\tLoss:" in line:
            assert line.split()[2] == "0.0983"
        if "Epoch: 8/1\\tLoss:" in line:
            assert line.split()[2] == "0.0796"
        if "Epoch: 9/1\\tLoss:" in line:
            assert line.split()[2] == "0.0653"
        if "Epoch: 10/1\\tLoss:" in line:
            assert line.split()[2] == "0.0560"

    # Checking if results are still the same
    cmd = ['./bin/kbp-cli.py',
           '--train', 'data/wn18/wordnet-mlj12-train.txt',
           '--lr', '0.1',
           '--model', 'TransE',
           '--similarity', 'l1',
           '--margin', '2',
           '--embedding-size', '50',
           '--nb-epochs', '5',
           '--clauses', 'data/wn18/clauses/clauses_0.9.pl',
           '--adv-weight', '1000',
           '--adv-lr', '0.1']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    for line in str(err).split("\\n"):
        if "Epoch: 1/1\\tLoss:" in line:
            assert line.split()[2] == "3.7271"
            assert line.split()[4] == "0.6187"
        if "Epoch: 2/1\\tLoss:" in line:
            assert line.split()[2] == "1.9572"
            assert line.split()[4] == "0.7526"
        if "Epoch: 3/1\\tLoss:" in line:
            assert line.split()[2] == "1.0932"
            assert line.split()[4] == "0.6586"
        if "Epoch: 4/1\\tLoss:" in line:
            assert line.split()[2] == "0.6326"
            assert line.split()[4] == "0.4441"
        if "Epoch: 5/1\\tLoss:" in line:
            assert line.split()[2] == "0.5513"

if __name__ == '__main__':
    pytest.main([__file__])
