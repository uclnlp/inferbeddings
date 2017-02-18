# Synthetic datasets for checking whether transitive axioms hold

```
$ yes p | head -n 8192 | awk '{ print "e" i++ "\t" $1 "\te" i "\n" "e" i++ "\t" $1 "\te" i++ "\n" }' > data.tsv
$ yes p | head -n 256 | awk '{ print "e" i++ "\t" $1 "\te" i "\n" "e" i++ "\t" $1 "\te" i++ "\n" }' > data-tiny.tsv

$ yes p | head -n 1024 | awk '{ print "e" i++ "\t" $1 "\te" ++i "\n" ; i++ }' > data-test.tsv
$ yes p | head -n 32 | awk '{ print "e" i++ "\t" $1 "\te" ++i "\n" ; i++ }' > data-test-tiny.tsv
```
