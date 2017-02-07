# FB15k - Rules

```bash
$ java -jar tools/amie_plus.jar data/fb15k/freebase_mtr100_mte100-train.txt > data/fb15k/rules/fb15k-rules.txt 2>&1
$ java -jar tools/amie_plus.jar -mins 1000 data/fb15k/freebase_mtr100_mte100-train.txt > data/fb15k/rules/fb15k-rules_mins=1000.txt 2>&1
$ java -jar tools/amie_plus.jar -mins 10000 data/fb15k/freebase_mtr100_mte100-train.txt > data/fb15k/rules/fb15k-rules_mins=10000.txt 2>&1
```
