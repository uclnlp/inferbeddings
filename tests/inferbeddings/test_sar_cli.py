# -*- coding: utf-8 -*-

import pytest
import subprocess

import sys
sys.setrecursionlimit(65535)


def test_sar_cli():
    # Checking if results are still not nice when not using adversarial training,
    # i.e. by using the parameters --adv-weight 0
    cmd = ['./bin/kbp-cli.py',
           '--train', 'data/synth/sar-small/data.tsv',
           '--test', 'data/synth/sar-small/data-test.tsv',
           '--clauses', 'data/synth/sar-small/clauses.pl',
           '--lr', '0.1',
           '--model', 'ComplEx',
           '--similarity', 'dot',
           '--margin', '1',
           '--embedding-size', '50',
           '--nb-epochs', '100',
           '--loss', 'hinge']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    # Hits@10 should be around 0% at first
    assert float(err.split()[-1][:-1]) < 60.0

    cmd = ['./bin/kbp-cli.py',
           '--train', 'data/synth/sar-small/data.tsv',
           '--test', 'data/synth/sar-small/data-test.tsv',
           '--clauses', 'data/synth/sar-small/clauses.pl',
           '--materialize', '--sar-weight', '0.0',
           '--lr', '0.1',
           '--model', 'ComplEx',
           '--similarity', 'dot',
           '--margin', '1',
           '--embedding-size', '50',
           '--nb-epochs', '100',
           '--loss', 'hinge']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    # Hits@10 should be around 0% at first
    assert float(err.split()[-1][:-1]) > 95.0

    cmd = ['./bin/kbp-cli.py',
           '--train', 'data/synth/sar-small/data.tsv',
           '--test', 'data/synth/sar-small/data-test.tsv',
           '--clauses', 'data/synth/sar-small/clauses.pl',
           '--sar-similarity', 'l2_sqr', '--sar-weight', '10.0',
           '--lr', '0.1',
           '--model', 'ComplEx',
           '--similarity', 'dot',
           '--margin', '1',
           '--embedding-size', '50',
           '--nb-epochs', '100',
           '--loss', 'hinge']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    # Hits@10 should be around 0% at first
    assert float(err.split()[-1][:-1]) > 95.0

if __name__ == '__main__':
    pytest.main([__file__])
