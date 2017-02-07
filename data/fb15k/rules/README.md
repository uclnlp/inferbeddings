# FB15k - Rules

```bash
$ java -jar tools/amie_plus.jar data/fb15k/freebase_mtr100_mte100-train.txt > data/fb15k/rules/fb15k-rules.txt 2>&1
$ java -jar tools/amie_plus.jar -mins 1000 data/fb15k/freebase_mtr100_mte100-train.txt > data/fb15k/rules/fb15k-rules_mins=1000.txt 2>&1
$ java -jar tools/amie_plus.jar -mins 10000 data/fb15k/freebase_mtr100_mte100-train.txt > data/fb15k/rules/fb15k-rules_mins=10000.txt 2>&1
$ java -jar tools/amie_plus.jar -mins 1000 -minis 1000 data/fb15k/freebase_mtr100_mte100-train.txt > data/fb15k/rules/fb15k-rules_mins=1000_minis=1000.txt 2>&1
$ java -jar tools/amie_plus.jar -mins 10000 -minis 10000 data/fb15k/freebase_mtr100_mte100-train.txt > data/fb15k/rules/fb15k-rules_mins=10000_minis=10000.txt 2>&1

$ wc -l data/fb15k/rules/*.txt
     652 data/fb15k/rules/fb15k-rules_mins=10000_minis=10000.txt
   41212 data/fb15k/rules/fb15k-rules_mins=10000.txt
    8925 data/fb15k/rules/fb15k-rules_mins=1000_minis=1000.txt
   41210 data/fb15k/rules/fb15k-rules_mins=1000.txt
   41211 data/fb15k/rules/fb15k-rules.txt
  133210 total

$ md5sum data/fb15k/rules/*.txt                                                                   
e7912de33c1306b2c5a790bf93603cc7  data/fb15k/rules/fb15k-rules_mins=10000_minis=10000.txt
24334fcda056b63086f283aeab75f639  data/fb15k/rules/fb15k-rules_mins=10000.txt
02bdb739e16e735b0c337a8ee4d4eccf  data/fb15k/rules/fb15k-rules_mins=1000_minis=1000.txt
bb264a73e30e85485d860790ddef40ee  data/fb15k/rules/fb15k-rules_mins=1000.txt
bd7de89cd25dc0d94fff511ff56d9569  data/fb15k/rules/fb15k-rules.txt

$ ./tools/amie-to-json.py data/fb15k/rules/fb15k-rules_mins=1000.txt > data/fb15k/rules/fb15k-rules_mins=1000.json
$ ./tools/amie-to-json.py data/fb15k/rules/fb15k-rules_mins=10000.txt > data/fb15k/rules/fb15k-rules_mins=10000.json
[..]
```
