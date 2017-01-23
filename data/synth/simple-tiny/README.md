# Synthetic datasets for checking whether simple axioms hold

```
$ yes | head -n 128 | awk '{ print "e" i++ "\tp\te" i++ }' > data.tsv
$ yes | head -n 1 | awk '{ print "q" i++ "\tq\tq" i++ }' >> data.tsv
```
