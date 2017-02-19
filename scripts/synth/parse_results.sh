#!/usr/bin/env bash

MODELS="DistMult TransE ComplEx"
TAGS="symm impl impl_inv impl_conj trans_single trans_diff multiple"
CONFS="0.3"
#CONFS="0.3 0.5 0.7"
#['exp_symm', 'exp_impl', 'exp_impl_inv', 'exp_impl_conj', 'exp_trans_single', 'exp_trans_diff']

#SUBSAMPLES="0.1 0.2 0.3 0.4 0.5 0.6 0.7 1"
SUBSAMPLES="0.5 1"
#SUBSAMPLES="0.1 0.5 1"

for TAG in $TAGS; do
    echo "EXPERIMENT $TAG"
    for CONF in $CONFS; do
        echo "    CLAUSE CONFIDENCE: ${CONF}"

        for MODEL in $MODELS; do
            echo "        (1) MODEL $MODEL"
            echo "            - without clauses:"
            for SS in $SUBSAMPLES; do
                BEST_AUCPR_VALID=`ls $@ | xargs grep "\[valid\]" | grep "AUC-PR" | grep "$TAG" | \
                    grep "adv_weight=0_" | grep "model=$MODEL" | \
                    grep "subsample_size=$SS" | grep "embedding_size=50" | grep "c$CONF" | \
                    awk '{ print $3 " " $0}' | sort -nr | head -n 1 | \
                    sed -e ''s/":INFO:"/" "/g'' | awk '{ print $2 }'`
                if [ !  -z  "$BEST_AUCPR_VALID" ]; then
                    AUCPR_TEST=`cat $BEST_AUCPR_VALID | grep "\[test\]" | grep "AUC-PR" | awk '{ print $3 }'`
                    echo "                subsample $SS: AUCPR = " `printf '%.*f' 2 $AUCPR_TEST` #"($BEST_AUCPR_VALID)"
                fi
            done

            echo "            - with clauses:"
            for SS in $SUBSAMPLES; do
                BEST_AUCPR_VALID=`ls $@ | xargs grep "\[valid\]" | grep "AUC-PR" | grep "$TAG" | \
                    grep -v "adv_weight=0_" | grep "model=$MODEL" | \
                    grep "subsample_size=$SS" | grep "embedding_size=50" | grep "c$CONF" | \
                    awk '{ print $3 " " $0}' | sort -nr | head -n 1 | \
                    sed -e ''s/":INFO:"/" "/g'' | awk '{ print $2 }'`
                if [ !  -z  "$BEST_AUCPR_VALID" ]; then
                    AUCPR_TEST=`cat $BEST_AUCPR_VALID | grep "\[test\]" | grep "AUC-PR" | awk '{ print $3 }'`
                    echo "                subsample $SS: AUCPR = " `printf '%.*f' 2 $AUCPR_TEST` #"($BEST_AUCPR_VALID)"
                fi
            done


        done
    done
done


