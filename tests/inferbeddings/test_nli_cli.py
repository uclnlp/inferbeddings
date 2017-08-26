# -*- coding: utf-8 -*-

import pytest
import subprocess

import sys
sys.setrecursionlimit(65535)


@pytest.mark.light
def test_nli_cli():
    cmd_str = './bin/nli-cli.py -f -n -m ff-dam --batch-size 32 --dropout-keep-prob 0.8 --representation-size 200 ' \
              '--optimizer adam --learning-rate 0.001 -c 100 -i normal --nb-epochs 50 --has-bos -t ' \
              'data/snli/tiny/tiny.jsonl.gz -v data/snli/tiny/tiny.jsonl.gz -T data/snli/tiny/tiny.jsonl.gz'
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
            assert '0.0316' in line
        if 'Epoch 20/1' in line:
            assert '0.0145' in line
        if 'Epoch 30/1' in line:
            assert '0.0105' in line
        if 'Epoch 40/1' in line:
            assert '0.0179' in line
        if 'Epoch 50/1' in line:
            assert '0.0008' in line

if __name__ == '__main__':
    pytest.main([__file__])
