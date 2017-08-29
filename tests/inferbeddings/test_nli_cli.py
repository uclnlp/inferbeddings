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
            assert '0.0304' in line
        if 'Epoch 20/1' in line:
            assert '0.0163' in line
        if 'Epoch 30/1' in line:
            assert '0.0112' in line
        if 'Epoch 40/1' in line:
            assert '0.0050' in line
        if 'Epoch 70/1' in line:
            assert '0.0044' in line
        if 'Epoch 80/1' in line:
            assert '0.0032' in line
        if 'Epoch 90/1' in line:
            assert '0.0003' in line
        if 'Epoch 100/1' in line:
            assert '0.0000' in line
if __name__ == '__main__':
    pytest.main([__file__])
