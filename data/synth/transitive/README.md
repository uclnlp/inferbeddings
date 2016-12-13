# Synthetic datasets for checking whether transitive axioms hold

```
$ yes p | head -n 8192 | awk '{ print "e" i++ "\t" $1 "\te" i "\n" "e" i++ "\t" $1 "\te" i++ "\n" }' > data.tsv
```
