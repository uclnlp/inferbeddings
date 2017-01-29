#!/usr/bin/env bash
ls $@ | wc -w

###
### MEAN RANK
###

BEST_RAW_MR_FILE=`ls $@ | xargs grep "MICRO (validation raw)" -A 3 | grep global | grep "hits@10:" | awk '{ print $6 " " $0 }' | sort -nr | sed -e ''s/"-INFO"/" "/g'' | tail -n 1 | awk '{ print $2 }'`
BEST_FILT_MR_FILE=`ls $@ | xargs grep "MICRO (validation filtered)" -A 3 | grep global | grep "hits@10:" | awk '{ print $6 " " $0 }' | sort -nr | sed -e ''s/"-INFO"/" "/g'' | tail -n 1 | awk '{ print $2 }'`

echo "Best MR, Raw:" $BEST_RAW_MR_FILE
echo "Best MR, Filt:" $BEST_FILT_MR_FILE

BEST_RAW_MR=`cat $BEST_RAW_MR_FILE | grep "MICRO (test raw)" -A 3 | grep global | grep "hits@10:" | awk '{ print $6 }' | tr -d ","`
BEST_FILT_MR=`cat $BEST_FILT_MR_FILE | grep "MICRO (test filtered)" -A 3 | grep global | grep "hits@10:" | awk '{ print $6 }' | tr -d ","`

echo "Test - Best Raw MR:" $BEST_RAW_MR
echo "Test - Best Filt MR:" $BEST_FILT_MR

echo

###
### MEAN RECIPROCAL RANK
###

BEST_RAW_MRR_FILE=`ls $@ | xargs grep "MICRO (validation raw)" -A 3 | grep global | grep "hits@10:" | awk '{ print $10 " " $0 }' | sort -nr | sed -e ''s/"-INFO"/" "/g'' | head -n 1 | awk '{ print $2 }'`
BEST_FILT_MRR_FILE=`ls $@ | xargs grep "MICRO (validation filtered)" -A 3 | grep global | grep "hits@10:" | awk '{ print $10 " " $0 }' | sort -nr | sed -e ''s/"-INFO"/" "/g'' | head -n 1 | awk '{ print $2 }'`

echo "Best MRR, Raw:" $BEST_RAW_MRR_FILE
echo "Best MRR, Filt:" $BEST_FILT_MRR_FILE

BEST_RAW_MRR=`cat $BEST_RAW_MRR_FILE | grep "MICRO (test raw)" -A 3 | grep global | grep "hits@10:" | awk '{ print $10 }' | tr -d ","`
BEST_FILT_MRR=`cat $BEST_FILT_MRR_FILE | grep "MICRO (test filtered)" -A 3 | grep global | grep "hits@10:" | awk '{ print $10 }' | tr -d ","`

echo "Test - Best Raw MRR:" $BEST_RAW_MRR
echo "Test - Best Filt MRR:" $BEST_FILT_MRR

echo

for N in 1 3 5 10
do
    ###
    ### HITS@N
    ###

    BEST_RAW_H_FILE=`ls $@ | xargs grep "MICRO (validation raw)" -A 3 | grep global | grep "hits@$N:" | awk '{ print $12 " " $0 }' | sort -nr | sed -e ''s/"-INFO"/" "/g'' | head -n 1 | awk '{ print $2 }'`
    BEST_FILT_H_FILE=`ls $@ | xargs grep "MICRO (validation filtered)" -A 3 | grep global | grep "hits@$N:" | awk '{ print $12 " " $0 }' | sort -nr | sed -e ''s/"-INFO"/" "/g'' | head -n 1 | awk '{ print $2 }'`

    echo "Best H@$N, Raw:" $BEST_RAW_H_FILE
    echo "Best H@$N, Filt:" $BEST_FILT_H_FILE

    BEST_RAW_H=`cat $BEST_RAW_H_FILE | grep "MICRO (test raw)" -A 3 | grep global | grep "hits@$N:" | awk '{ print $12 }' | tr -d ","`
    BEST_FILT_H=`cat $BEST_FILT_H_FILE | grep "MICRO (test filtered)" -A 3 | grep global | grep "hits@$N:" | awk '{ print $12 }' | tr -d ","`

    echo "Test - Best Raw Hits@$N:" $BEST_RAW_H
    echo "Test - Best Filt Hits@$N:" $BEST_FILT_H

    echo
done
