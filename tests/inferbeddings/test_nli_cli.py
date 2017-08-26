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

    # p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # out, err = p.communicate()

    # print(err)
    # print(err.split()[-1][:-1])

    cmd_str = './bin/nli-cli.py -f -n -m ff-dam --batch-size 32 --dropout-keep-prob 0.8 ' \
              '--representation-size 200 --optimizer adam --learning-rate 0.001 -c 100 ' \
              '-i normal --nb-epochs 50 -t data/snli/tiny/tiny.jsonl.gz -v data/snli/tiny/tiny.jsonl.gz ' \
              '-T data/snli/tiny/tiny.jsonl.gz'
    cmd = cmd_str.split()

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    sys.stdout = sys.stderr

    lines = err.decode("utf-8").split("\n")

    for line in lines:
        print(line)

        if 'Epoch 1	Loss:' in line:
            assert 'Loss: 1.0988' in line
        if 'Epoch 2	Loss:' in line:
            assert 'Loss: 1.0986' in line
        if 'Epoch 3	Loss:' in line:
            assert 'Loss: 1.0986' in line
        if 'Epoch 4	Loss:' in line:
            assert 'Loss: 1.0986' in line
        if 'Epoch 5	Loss:' in line:
            assert 'Loss: 1.0983' in line
        if 'Epoch 6	Loss:' in line:
            assert 'Loss: 1.0984' in line
        if 'Epoch 7 Loss:' in line:
            assert 'Loss: 1.0941' in line
        if 'Epoch 8 Loss:' in line:
            assert 'Loss: 1.0876' in line
        if 'Epoch 9 Loss:' in line:
            assert 'Loss: 1.0632' in line
        if 'Epoch 10    Loss:' in line:
            assert 'Loss: 1.0202' in line
        if 'Epoch 30    Loss:' in line:
            assert 'Loss: 0.1977' in line
        if 'Epoch 40    Loss:' in line:
            assert 'Loss: 0.0087' in line

if __name__ == '__main__':
    pytest.main([__file__])
