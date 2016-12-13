# Synthetic datasets for checking whether symmetry axioms hold

```
$ yes p | head -n 8192 | awk '{ print "e" i++ "\t" $1 "\te" i++ }' > data.tsv
```
