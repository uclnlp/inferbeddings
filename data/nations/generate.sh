#!/bin/bash

cat nat_fix01.db  | grep -v "RR(" | grep "R(" | tr "(" "\t" | tr "," "\t" | tr -d ")" | awk '{ print $2 " \"" $1 "\" " $3 }' > nations.tsv
cat nat_fix01.db | grep "RR(" | tr "," "\t" | tr "(" "\t" | tr -d ")" | awk '{ print $3 " \"" $2 "\" " $4 }' >> nations.tsv

cat nat_fix01.db | grep "RR(" | tr "," "\t" | tr "(" "\t" | tr -d ")" | awk '{ print $3 " \"" $2 "\" " $4 }' > nations_ternary.tsv
