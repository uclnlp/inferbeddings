#!/usr/bin/env bash

ls *.log | xargs grep "\[test\]" | sed -e ''s/":INFO:"/" "/g'' | grep "AUC-PR" | awk '{ print $4 "\t" $1 }'