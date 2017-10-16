# Generating new datasets

```bash
$ ./tools/contradictions.py ../snli_1.0_dev.jsonl.gz > snli_1.0_contradictions_dev.json
$ ./tools/contradictions.py ../snli_1.0_test.jsonl.gz > snli_1.0_contradictions_test.json
$ ./tools/contradictions.py ../snli_1.0_dev.jsonl.gz -i > snli_1.0_contradictions_inverse_dev.json
$ ./tools/contradictions.py ../snli_1.0_test.jsonl.gz -i > snli_1.0_contradictions_inverse_test.json
$ md5sum *.json
3f9ccd025e423adde20e8cbca79ec968  snli_1.0_contradictions_dev.json
3504039e23aa1bbf222606e125537489  snli_1.0_contradictions_inverse_dev.json
2f60b975d0d8509a82eaeffa652ff6a5  snli_1.0_contradictions_inverse_test.json
839a766dbf34ef7cf6ab910679678294  snli_1.0_contradictions_test.json
$ gzip -9 *.json
```
