#!/usr/bin/env bash

MODELS="DistMult TransE ComplEx"

#without clauses
for MODEL in $MODELS
do
    BEST_AUCPR_FILE=`ls $@ | xargs grep "\[valid\]" | grep $MODEL | grep "AUC-PR:" | grep -v 'clauses_' | awk '{ print $3 " " $0}' | sort -nr | head -n 1 | sed -e ''s/":INFO:"/" "/g'' | awk '{ print $2 }'`
    if [ -n "$BEST_AUCPR_FILE" ]
    then
        BEST_AUCPR_VALID=`cat $BEST_AUCPR_FILE | grep '\[valid\]' | grep 'AUC-PR' | awk '{ print $3 }'`
#        echo "Valid - Best AUC-PR ($MODEL, without clauses):" $BEST_AUCPR_VALID #"($BEST_AUCPR_FILE)"
        AUCPR_TEST=`cat $BEST_AUCPR_FILE | grep '\[test\]' | grep 'AUC-PR' | awk '{ print $3 }'`
        echo "Test data (best validation AUC-PR for $MODEL, without clauses): AUCPR = " $AUCPR_TEST #"($BEST_AUCPR_FILE)"
    fi
done

#with clauses
for MODEL in $MODELS
do
    BEST_AUCPR_FILE=`ls $@ | xargs grep "\[valid\]" | grep $MODEL | grep "AUC-PR:" | grep 'clauses_' | awk '{ print $3 " " $0}' | sort -nr | head -n 1 | sed -e ''s/":INFO:"/" "/g'' | awk '{ print $2 }'`
    if [ -n "$BEST_AUCPR_FILE" ]
    then
        BEST_AUCPR_VALID=`cat $BEST_AUCPR_FILE | grep '\[valid\]' | grep 'AUC-PR' | awk '{ print $3 }'`
#        echo "Valid - Best AUC-PR ($MODEL, with clauses):" $BEST_AUCPR_VALID #"($BEST_AUCPR_FILE)"
        AUCPR_TEST=`cat $BEST_AUCPR_FILE | grep '\[test\]' | grep 'AUC-PR' | awk '{ print $3 }'`
        echo "Test data (best validation AUC-PR for $MODEL, with clauses): AUCPR = " $AUCPR_TEST #"($BEST_AUCPR_FILE)"
    fi
done
