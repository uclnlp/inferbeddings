#!/bin/bash

zcat ../yago3_mte10_5k/tmp/yagoFacts.tsv.gz | tail -n +2 | awk 'BEGIN {FS="\t"} ; { print $2 "\t" $3 "\t" $4 }' | tr -d '>' | tr -d '<' > yago3.tsv
gzip -9 yago3.tsv
mkdir stats
zcat yago3.tsv.gz | awk '{ print $1 "\n" $3 }' | sort | uniq -c | awk '{if ($1 >= 100) {print $2}}' > stats/yago3_entities_mte100.txt
./tools/filter.py yago3.tsv.gz stats/yago3_entities_mte100.txt > yago3_mte100.tsv
gzip -9 yago3_mte100.tsv
./tools/split.py yago3_mte100.tsv.gz --train yago3_mte100-train.tsv --valid yago3_mte100-valid.tsv --valid-size 5000 --test yago3_mte100-test.tsv --test-size 5000 --seed 0
