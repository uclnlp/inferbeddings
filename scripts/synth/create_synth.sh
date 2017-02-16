#!/usr/bin/env bash

TAG="exp_symm"
DIR="../../data/synth/sampled"
python3 sample_kb.py \
                    --entities 30 \
                    --predicates 10 \
                    --test-prob 0.7 \
                    --arg-density 0.1 \
                    --fact-prob 0.1 \
                    --symm 5 \
                    --impl 0 \
                    --impl-inv 0 \
                    --impl-conj 0 \
                    --trans-single 0 \
                    --trans-diff 0 \
                    --tag $TAG \
                    --dir $DIR

