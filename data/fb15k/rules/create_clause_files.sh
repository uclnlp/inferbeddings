#!/usr/bin/env bash

A2C="../../../tools/amie-to-clauses.py"
RULEFILE="fb15k-rules.txt"
CLAUSEFILE="../clauses/clauses_"

#different types of rule files  ([L]ow/[H]igh [C]onfidence/[S]upport)
LC_HS="${CLAUSEFILE}lowconf_highsupp.pl"
LC_LS="${CLAUSEFILE}lowconf_lowsupp.pl"
HC_HS="${CLAUSEFILE}highconf_highsupp.pl"
HC_LS="${CLAUSEFILE}highconf_lowsupp.pl"

LC=0.85
HC=0.99
LS=50
HS=1000

#generate rule files
$A2C $RULEFILE -C $LC -B $HS > $LC_HS
$A2C $RULEFILE -C $LC -B $LS > $LC_LS
$A2C $RULEFILE -C $HC -B $HS > $HC_HS
$A2C $RULEFILE -C $HC -B $LS > $HC_LS

#count rules
echo "$LC_HS : $(cat $LC_HS | wc -l) rules"
echo "$LC_LS : $(cat $LC_LS | wc -l) rules"
echo "$HC_HS : $(cat $HC_HS | wc -l) rules"
echo "$HC_LS : $(cat $HC_LS | wc -l) rules"
