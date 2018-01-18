# Subsampling in SNLI

```shell
$ ./data/snli/tools/sample.py -p data/snli/snli_1.0_train.jsonl.gz -f 0.1 > data/snli/fractions/snli_1.0_train_0.1.jsonl
550152it [00:03, 151803.89it/s]
$ ./data/snli/tools/sample.py -p data/snli/snli_1.0_train.jsonl.gz -f 0.2 > data/snli/fractions/snli_1.0_train_0.2.jsonl
550152it [00:03, 156790.81it/s]
$ ./data/snli/tools/sample.py -p data/snli/snli_1.0_train.jsonl.gz -f 0.5 > data/snli/fractions/snli_1.0_train_0.5.jsonl
550152it [00:03, 157111.60it/s]
$ md5sum data/snli/fractions/*.jsonl
768483d86c115aadcac8a0faaae402ca  data/snli/fractions/snli_1.0_train_0.1.jsonl
7659299ada7ad90c846982f41b25a3c9  data/snli/fractions/snli_1.0_train_0.2.jsonl
56f5687ab350481d6b4d24b70278921a  data/snli/fractions/snli_1.0_train_0.5.jsonl
$ gzip -9 data/snli/fractions/*.jsonl
$ du -hs data/snli/fractions/*.gz
8.3M    data/snli/fractions/snli_1.0_train_0.1.jsonl.gz
17M     data/snli/fractions/snli_1.0_train_0.2.jsonl.gz
42M     data/snli/fractions/snli_1.0_train_0.5.jsonl.gz
$
```
