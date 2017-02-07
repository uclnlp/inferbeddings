# FB15k - Clauses

```bash
$ ./tools/amie-to-clauses.py data/fb15k/rules/fb15k-rules.txt -t 0.9 > data/fb15k/clauses/clauses_0.9.pl
$ ./tools/amie-to-clauses.py data/fb15k/rules/fb15k-rules.txt -t 0.99 > data/fb15k/clauses/clauses_0.99.pl
$ ./tools/amie-to-clauses.py data/fb15k/rules/fb15k-rules.txt -t 0.999 > data/fb15k/clauses/clauses_0.999.pl

$ ./tools/amie-to-clauses.py data/fb15k/rules/fb15k-rules_mins=1000_minis=1000.txt -t 0.9 > data/fb15k/clauses/clauses_0.9_mins=1000_minis=1000.pl
$ ./tools/amie-to-clauses.py data/fb15k/rules/fb15k-rules_mins=1000_minis=1000.txt -t 0.99 > data/fb15k/clauses/clauses_0.99_mins=1000_minis=1000.pl
$ ./tools/amie-to-clauses.py data/fb15k/rules/fb15k-rules_mins=1000_minis=1000.txt -t 0.999 > data/fb15k/clauses/clauses_0.999_mins=1000_minis=1000.pl
$ ./tools/amie-to-clauses.py data/fb15k/rules/fb15k-rules_mins=10000_minis=10000.txt -t 0.9 > data/fb15k/clauses/clauses_0.9_mins=10000_minis=10000.pl
$ ./tools/amie-to-clauses.py data/fb15k/rules/fb15k-rules_mins=10000_minis=10000.txt -t 0.99 > data/fb15k/clauses/clauses_0.99_mins=10000_minis=10000.pl
$ ./tools/amie-to-clauses.py data/fb15k/rules/fb15k-rules_mins=10000_minis=10000.txt -t 0.999 > data/fb15k/clauses/clauses_0.999_mins=10000_minis=10000.pl
```
