#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

import sys
import os

import logging
logger = logging.getLogger(os.path.basename(sys.argv[0]))


def clean(x):
    return x.replace(' ', '_').replace('(', '').replace(')', '').lower()

random.seed(0)

acronym2name = {}
table = []

countries, regions, subregions = set(), set(), set()
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


def write_to_file(path, instances):
    with open(path, 'w') as f:
        for instance in instances:
            f.write(instance)

write_to_file('countries.tsv', sorted(facts))
write_to_file('data/countries.lst', sorted(countries))
write_to_file('data/regions.lst', sorted(regions))
write_to_file('data/subregions.lst', sorted(subregions))

countries = list(countries)

countries_w_neighbors = []
countries_wo_neigbors = []

for country in countries:
    if len(country2neighbors[country]) > 0:
        countries_w_neighbors.append(country)
    else:
        countries_wo_neigbors.append(country)

num_countries = len(countries)
splits = [0.1, 0.1, 0.8]
splits = [int(num_countries * x) for x in splits]

train_countries = countries_wo_neigbors
random.shuffle(countries_w_neighbors)

test = countries_w_neighbors[0:splits[0]]
dev = countries_w_neighbors[splits[0]:splits[0]+splits[1]]
train = countries_w_neighbors[splits[0]+splits[1]:] + countries_wo_neigbors


def ensure_consistency(train, dev, test, tries=100):
    logger.info("Ensuring consistency", tries)
    swap_test_with_train = []
    swap_dev_with_train = []
    new_test = []
    new_dev = []

    for x in test:
        neighbors = country2neighbors[x]
        train_neighbors = [x for x in neighbors if x in train]
        if len(train_neighbors) == 0:
            # print(x, len(neighbors), len(train_neighbors))
            swap_test_with_train.append(x)
        else:
            new_test.append(x)

    for x in dev:
        neighbors = country2neighbors[x]
        train_neighbors = [x for x in neighbors if x in train]
        if len(train_neighbors) == 0:
            # print(x, len(neighbors), len(train_neighbors))
            swap_dev_with_train.append(x)
        else:
            new_dev.append(x)

    if len(swap_dev_with_train) + len(swap_test_with_train) == 0:
        return train, dev, test
    elif tries == 0:
        logger.warning("Damn!")
    else:
        random.shuffle(train)
        new_dev += train[len(swap_test_with_train):len(swap_test_with_train) +
                                                   len(swap_dev_with_train)]
        new_test += train[:len(swap_test_with_train)]
        train = train[len(swap_test_with_train)+len(swap_dev_with_train):] + swap_test_with_train + swap_dev_with_train
        return ensure_consistency(train, new_dev, new_test, tries - 1)


train, dev, test = ensure_consistency(train, dev, test, tries=10)

with open("./s/test.txt", "w") as f_out:
    for country in test:
        f_out.write("%s\n" % country)
    f_out.close()


with open("./s/dev.txt", "w") as f_out:
    for country in dev:
        f_out.write("%s\n" % country)
    f_out.close()

with open("./s/countries_S1.nl", "w") as f_out:
    for corpus in [dev, test]:
        for country in corpus:
            for fact in country2facts[country]:
                arg2 = fact.split(",")[1].split(")")[0]
                if arg2 not in regions:
                    f_out.write(fact)
    for country in train:
        for fact in country2facts[country]:
            f_out.write(fact)

    f_out.close()

with open("./s/countries_S2.nl", "w") as f_out:
    for corpus in [dev, test]:
        for country in corpus:
            for fact in country2facts[country]:
                arg2 = fact.split(",")[1].split(")")[0]
                if arg2 not in regions and arg2 not in subregions:
                    f_out.write(fact)
    for country in train:
        for fact in country2facts[country]:
            f_out.write(fact)

    f_out.close()

with open("./s/countries_S3.nl", "w") as f_out:
    for corpus in [dev, test]:
        for country in corpus:
            for fact in country2facts[country]:
                arg2 = fact.split(",")[1].split(")")[0]
                if arg2 not in regions and arg2 not in subregions:
                    f_out.write(fact)

    test_countries = set(dev).union(set(test))
    for country in train:
        is_neighbor_of_test_country = False
        country_neighbors = country2neighbors[country]
        country_test_neighbors = \
            test_countries.intersection(set(country_neighbors))
        is_test_neighbor = len(country_test_neighbors) > 0

        for fact in country2facts[country]:
            arg2 = fact.split(",")[1].split(")")[0]
            if arg2 not in regions or not is_test_neighbor:
                f_out.write(fact)

    f_out.close()
