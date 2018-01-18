# Subsampling in SNLI

```shell
$ ./data/snli/tools/sample.py -p data/snli/snli_1.0_train.jsonl.gz -f 0.1 > data/snli/fractions/snli_1.0_train_0.1.jsonl                                            
550152it [00:10, 52129.59it/s]
$ ./data/snli/tools/sample.py -p data/snli/snli_1.0_train.jsonl.gz -f 0.2 > data/snli/fractions/snli_1.0_train_0.2.jsonl                                                
550152it [00:10, 52695.59it/s]
$ ./data/snli/tools/sample.py -p data/snli/snli_1.0_train.jsonl.gz -f 0.5 > data/snli/fractions/snli_1.0_train_0.5.jsonl                                                
550152it [00:09, 55799.06it/s]
$ md5sum data/snli/fractions/*.jsonl
7e90da59b988d51a3de9413574626314  data/snli/fractions/snli_1.0_train_0.1.jsonl
7cf835b95c1bf457ca9b8a514f7cfdb9  data/snli/fractions/snli_1.0_train_0.2.jsonl
4c378b2f4fcd27c5a5398323bc244b3f  data/snli/fractions/snli_1.0_train_0.5.jsonl
$ gzip -9 data/snli/fractions/*.jsonl
$ du -hs data/snli/fractions/*.gz
8.3M    data/snli/fractions/snli_1.0_train_0.1.jsonl.gz
17M     data/snli/fractions/snli_1.0_train_0.2.jsonl.gz
42M     data/snli/fractions/snli_1.0_train_0.5.jsonl.gz
$
```
