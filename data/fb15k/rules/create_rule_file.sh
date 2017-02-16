#!/usr/bin/env bash

AMIE="../../../tools/amie_plus.jar"
TRAIN="../freebase_mtr100_mte100-train.txt"

RULEFILE="fb15k-rules-train.txt"

java -jar $AMIE -full -minhc 0.001 -minis 10 -mins 10 -pm support -verbose $TRAIN > $RULEFILE