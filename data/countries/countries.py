#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

import sys
import os

import logging
logger = logging.getLogger(os.path.basename(sys.argv[0]))


def clean(x):
    return x.replace(' ', '_').replace('(', '').replace(')', '').lower()


acronym2name = {}
table = []

countries = set()
regions = set()
subregions = set()

country2neighbors = {}

with open('data/countries.csv', 'r') as f_in:
    for line in f_in.readlines()[1:]:
        line = line.strip().split(';')
        country = clean(line[0][1:-1].split(',')[0])
        acronym = line[4][1:-1]
        acronym2name[acronym] = country
        capital = line[8]
        region = clean(line[10][1:-1])
        subregion = clean(line[11][1:-1])
        borders = line[17][1:-1].split(',')
        if borders == ['']:
            borders = []

        assert country != ''

        if region:
            regions.add(region)
        if subregion:
            subregions.add(subregion)

        if region and subregion:
            table.append((country, region, subregion, borders))
            countries.add(country)

facts = set()

country2facts = {}

for country, region, subregion, borders in table:
    country_facts = [
        '{}\tlocatedIn\t{}\n'.format(country, region),
        '{}\tlocatedIn\t{}\n'.format(country, subregion),
        '{}\tlocatedIn\t{}\n'.format(subregion, region)
    ]
    neighbors = [
        '{}\tneighborOf\t{}\n'.format(country, acronym2name[x]) for x in borders
    ]
    country2neighbors[country] = set(acronym2name[x] for x in borders)
    for neighbor in neighbors:
        country_facts.append(neighbor)
    for fact in country_facts:
        facts.add(fact)
    country2facts[country] = country_facts

assert len(countries) == 244
assert len(regions) == 5
assert len(subregions) == 23

with open('countries.tsv', 'w') as f_out:
    for fact in sorted(facts):
        f_out.write(fact)

with open('data/countries.lst', 'w') as f_out:
    for country in sorted(countries):
        f_out.write('{}\n'.format(country))

with open('data/regions.lst', 'w') as f_out:
    for region in sorted(regions):
        f_out.write('{}\n'.format(region))

with open('data/subregions.lst', 'w') as f_out:
    for region in sorted(subregions):
        f_out.write('{}\n'.format(region))
