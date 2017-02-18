# Synthetic datasets for showing what happens at embeddings level, with symmetric axioms

```
$ yes | head -n 64 | awk '{ print "e" i++ "\tp\te" i++ }' > data.tsv

$ cat data.tsv | awk '{ print $3 "\tq\t" $1 }' > data-inverse-test.tsv
```
