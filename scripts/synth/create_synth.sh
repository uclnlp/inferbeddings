#!/usr/bin/env bash

DIR="../../data/synth/sampled_small"

ENT=30
PRED=15
RULES=10
FACTPROB=0.1
ARGDENSITY=0.1

CONFS="0.0 0.2 0.4 0.6"

SEEDS="0 1 2 3 4 5 6 7 8 9"

for SEED in $SEEDS; do
    for CONF in $CONFS; do
        TESTPROB=`echo "1.0-$CONF" | bc -l`

        TAG="symm"
        python3 sample_kb.py \
                            --entities $ENT \
                            --predicates $PRED \
                            --test-prob $TESTPROB \
                            --arg-density $ARGDENSITY \
                            --fact-prob $FACTPROB \
                            --symm $RULES \
                            --impl 0 \
                            --impl-inv 0 \
                            --impl-conj 0 \
                            --trans-single 0 \
                            --trans-diff 0 \
                            --tag "${TAG}_c${CONF}_v${SEED}" \
                            --seed ${SEED} \
                            --dir $DIR

        TAG="impl"
        python3 sample_kb.py \
                            --entities $ENT \
                            --predicates $PRED \
                            --test-prob $TESTPROB \
                            --arg-density $ARGDENSITY \
                            --fact-prob $FACTPROB \
                            --symm 0 \
                            --impl $RULES \
                            --impl-inv 0 \
                            --impl-conj 0 \
                            --trans-single 0 \
                            --trans-diff 0 \
                            --tag "${TAG}_c${CONF}_v${SEED}" \
                            --seed ${SEED} \
                            --dir $DIR

        TAG="impl_inv"
        python3 sample_kb.py \
                            --entities $ENT \
                            --predicates $PRED \
                            --test-prob $TESTPROB \
                            --arg-density $ARGDENSITY \
                            --fact-prob $FACTPROB \
                            --symm 0 \
                            --impl 0 \
                            --impl-inv $RULES \
                            --impl-conj 0 \
                            --trans-single 0 \
                            --trans-diff 0 \
                            --tag "${TAG}_c${CONF}_v${SEED}" \
                            --seed ${SEED} \
                            --dir $DIR

    #    TAG="impl_conj"
    #    python3 sample_kb.py \
    #                        --entities $ENT \
    #                        --predicates $PRED \
    #                        --test-prob $TESTPROB \
    #                        --arg-density $ARGDENSITY \
    #                        --fact-prob $FACTPROB \
    #                        --symm 0 \
    #                        --impl 0 \
    #                        --impl-inv 0 \
    #                        --impl-conj $RULES \
    #                        --trans-single 0 \
    #                        --trans-diff 0 \
    #                        --tag "${TAG}_c${CONF}_v${SEED}" \
    #                        --seed ${SEED} \
    #                        --dir $DIR

        TAG="trans_single"
        python3 sample_kb.py \
                            --entities $ENT \
                            --predicates $PRED \
                            --test-prob $TESTPROB \
                            --arg-density $ARGDENSITY \
                            --fact-prob $FACTPROB \
                            --symm 0 \
                            --impl 0 \
                            --impl-inv 0 \
                            --impl-conj 0 \
                            --trans-single $RULES \
                            --trans-diff 0 \
                            --tag "${TAG}_c${CONF}_v${SEED}" \
                            --seed ${SEED} \
                            --dir $DIR

        TAG="trans_diff"
        python3 sample_kb.py \
                            --entities $ENT \
                            --predicates $PRED \
                            --test-prob $TESTPROB \
                            --arg-density $ARGDENSITY \
                            --fact-prob $FACTPROB \
                            --symm 0 \
                            --impl 0 \
                            --impl-inv 0 \
                            --impl-conj 0 \
                            --trans-single 0 \
                            --trans-diff $RULES \
                            --tag "${TAG}_c${CONF}_v${SEED}" \
                            --seed ${SEED} \
                            --dir $DIR

    #    TAG="multiple"
    #    python3 sample_kb.py \
    #                        --entities $ENT \
    #                        --predicates $PRED \
    #                        --test-prob $TESTPROB \
    #                        --arg-density $ARGDENSITY \
    #                        --fact-prob $FACTPROB \
    #                        --symm 2 \
    #                        --impl 2 \
    #                        --impl-inv 2 \
    #                        --impl-conj 2 \
    #                        --trans-single 2 \
    #                        --trans-diff 2 \
    #                        --tag "${TAG}_c${CONF}_v${SEED}" \
    #                        --seed ${SEED} \
    #                        --dir $DIR


    done
done
