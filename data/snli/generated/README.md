# Generating new datasets

```bash
$ ./tools/contradictions.py ../snli_1.0_dev.jsonl.gz > snli_1.0_contradictions_dev.json
$ ./tools/contradictions.py ../snli_1.0_test.jsonl.gz > snli_1.0_contradictions_test.json
$ ./tools/contradictions.py ../snli_1.0_dev.jsonl.gz -i > snli_1.0_contradictions_inverse_dev.json
$ ./tools/contradictions.py ../snli_1.0_test.jsonl.gz -i > snli_1.0_contradictions_inverse_test.json
$ md5sum *.json
84fcfe7fc53052d0a0eee6707a0c2fc3  snli_1.0_contradictions_dev.json
6052bf57de4c3fc8d0da02c175755263  snli_1.0_contradictions_inverse_dev.json
f3b85bfc9046c343ac8f0fdb1ad6906e  snli_1.0_contradictions_inverse_test.json
95618fab4a3044f055d557c8ece23aa1  snli_1.0_contradictions_test.json
$ gzip -9 *.json
```
