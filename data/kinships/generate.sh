#!/usr/bin/env bash
cat kin.db  | tr "(" "\t" | tr "," "\t" | tr -d ")" | awk '{ print $3 " " tolower($2) " " $4 }' > kinships.tsv
python3 ./make_folds_stratified.py kinships.tsv
for f in `find ./stratified_folds -name '*.tsv'`;do gzip -k $f; done