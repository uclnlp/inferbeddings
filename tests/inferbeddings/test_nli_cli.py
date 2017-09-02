# -*- coding: utf-8 -*-

import pytest
import subprocess

import sys
sys.setrecursionlimit(65535)


@pytest.mark.light
def test_nli_cli():
    cmd_str = './bin/nli-cli.py -f -n -m ff-dam --batch-size 32 --dropout-keep-prob 0.8 --representation-size 200 ' \
              '--optimizer adam --learning-rate 0.001 -c 100 -i normal --nb-epochs 100 --has-bos -t ' \
              'data/snli/tiny/tiny.jsonl.gz -v data/snli/tiny/tiny.jsonl.gz -T data/snli/tiny/tiny.jsonl.gz -r 999999'
    cmd = cmd_str.split()

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    sys.stdout = sys.stderr

    lines = err.decode("utf-8").split("\n")

    for line in lines:
        print(line)

        if 'Epoch 1/1' in line:
            assert '0.0343' in line
        if 'Epoch 10/1' in line:
            assert '0.0330' in line
        if 'Epoch 20/1' in line:
            assert '0.0217' in line
        if 'Epoch 30/1' in line:
            assert '0.0150' in line
        if 'Epoch 40/1' in line:
            assert '0.0122' in line
        if 'Epoch 70/1' in line:
            assert '0.0072' in line
        if 'Epoch 80/1' in line:
            assert '0.0033' in line
        if 'Epoch 90/1' in line:
            assert '0.0027' in line
        if 'Epoch 100/1' in line:
            assert '0.0008' in line


@pytest.mark.light
def test_nli_cli_restore():
    cmd_str = './bin/nli-cli.py -f -n -m ff-dam --batch-size 32 --dropout-keep-prob 0.8 ' \
              '--representation-size 200 --optimizer adam --learning-rate 0.0 -c 100 -i normal ' \
              '--nb-epochs 1 --has-bos -t data/snli/tiny/tiny.jsonl.gz -v data/snli/tiny/tiny.jsonl.gz ' \
              '-T data/snli/tiny/tiny.jsonl.gz --restore tests/models/snly_tiny_ffdam/dam_1 --report 1'
    cmd = cmd_str.split()

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    sys.stdout = sys.stderr

    lines = err.decode("utf-8").split("\n")

    for line in lines:
        print(line)

        if 'Dev Acc' in line:
            assert 'Dev Acc: 100.00' in line
            assert 'Test Acc: 100.00' in line

if __name__ == '__main__':
    pytest.main([__file__])
