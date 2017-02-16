#!/usr/bin/env bash

A2C="../../../tools/amie-to-clauses.py"
RULEFILE="fb15k-rules-train-highrecall.txt"
CLAUSEFILE="../clauses/clauses_"

minSupport=200

#generate rule files
for conf in 0.85 0.95 0.99; do
    cfile="${CLAUSEFILE}conf_${conf}.pl"
    $A2C $RULEFILE -C $conf -B $minSupport > $cfile
    echo "$cfile : $(cat $cfile | wc -l) rules"
done
