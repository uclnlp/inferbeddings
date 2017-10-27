#!/usr/bin/env bash

./tools/contradictions-cli.py ../snli_1.0_dev.jsonl.gz > snli_1.0_contradictions_dev.json
./tools/contradictions-cli.py ../snli_1.0_test.jsonl.gz > snli_1.0_contradictions_test.json
./tools/contradictions-cli.py ../snli_1.0_dev.jsonl.gz -i > snli_1.0_contradictions_inverse_dev.json
./tools/contradictions-cli.py ../snli_1.0_test.jsonl.gz -i > snli_1.0_contradictions_inverse_test.json

./tools/contradictions-cli.py ../snli_1.0_dev.jsonl.gz -r > snli_1.0_all_contradictions_dev.json
./tools/contradictions-cli.py ../snli_1.0_test.jsonl.gz -r > snli_1.0_all_contradictions_test.json
./tools/contradictions-cli.py ../snli_1.0_dev.jsonl.gz -i -r > snli_1.0_all_contradictions_inverse_dev.json
./tools/contradictions-cli.py ../snli_1.0_test.jsonl.gz -i -r > snli_1.0_all_contradictions_inverse_test.json


md5sum *.json
