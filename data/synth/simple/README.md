# Synthetic datasets for checking whether simple axioms hold

```
$ yes p | head -n 8192 | awk '{ print "e" i++ "\t" $1 "\te" i++ }' > data.tsv
$ yes q | head -n 1 | awk '{ print "q" i++ "\t" $1 "\tq" i++ }' >> data.tsv
```
