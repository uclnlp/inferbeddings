#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

from typing import NamedTuple, List
import json

from sklearn.model_selection import train_test_split

import logging
logger = logging.getLogger(os.path.basename(sys.argv[0]))


def norm(name):
    return name.replace(' ', '_').replace("'", '').lower()


def write_to_file(path, instances):
    with open(path, 'w') as f:
        for instance in instances:
            f.write(instance)


def write_triples_to_file(path, triples):
    with open(path, 'w') as f:
        for s, p, o in triples:
            f.write('{}\t{}\t{}\n'.format(s, p, o))


def main(argv):
    Country = NamedTuple('Country', [('name', str), ('region', str), ('subregion', str), ('neighbors', List[str])])
    code_to_country, country_name_to_country = dict(), dict()

    with open('countries.json', 'r') as fp:
        countries = json.load(fp)

    for c in countries:
        country = Country(norm(c['name']['official']), c['region'], c['subregion'], c['borders'])
        for code in {c['cca2'], c['ccn3'], c['cca3']}:
            code_to_country[code] = country
            country_name_to_country[norm(c['name']['official'])] = country

    triples = set()
    country_names, region_names, subregion_names = set(), set(), set()

    for c in countries:
        if len(c['region']) > 0:
            triples |= {(norm(c['name']['official']), 'locatedIn', norm(c['region']))}

        if len(c['subregion']) > 0:
            triples |= {(norm(c['region']), 'locatedIn', norm(c['subregion']))}
            triples |= {(norm(c['name']['official']), 'locatedIn', norm(c['subregion']))}

        for border in c['borders']:
            neighbor_name = code_to_country[border].name
            triples |= {(norm(c['name']['official']), 'neighborOf', neighbor_name)}

        country_names |= {norm(c['name']['official'])}

        if len(c['region']) > 0:
            region_names |= {norm(c['region'])}

        if len(c['subregion']) > 0:
            subregion_names |= {norm(c['subregion'])}

    assert len(country_names) == 248
    assert len(region_names) == 5
    assert len(subregion_names) == 23

    if not os.path.exists('data'):
        os.makedirs('data')

    write_to_file('data/countries.lst', sorted(country_names))
    write_to_file('data/regions.lst', sorted(region_names))
    write_to_file('data/subregions.lst', sorted(subregion_names))

    def is_consistent(_train, _test):
        for test_country_name in _test:
            _is_consistent = False
            for neighbor_code in country_name_to_country[test_country_name].neighbors:
                _neighbor_name = code_to_country[neighbor_code].name
                _is_consistent = _is_consistent or _neighbor_name in _train
            if not _is_consistent:
                return False
        return True

    country_names_lst = sorted(country_names)
    consistent, seed = None, 0

    while consistent is None:
        logging.debug('Trying seed {} ..'.format(seed))
        train, valid_test = train_test_split(country_names_lst, train_size=0.8, random_state=seed)
        valid, test = train_test_split(valid_test, train_size=0.5, random_state=seed)

        if is_consistent(train, test):
            consistent = (train, valid, test)
        seed += 1

    train, valid, test = consistent
    print(len(train), len(valid), len(test))

    write_to_file('./countries_train.lst', sorted(train))
    write_to_file('./countries_valid.lst', sorted(valid))
    write_to_file('./countries_test.lst', sorted(test))

    if not os.path.exists('s1'):
        os.makedirs('s1')

    s1_triples = set()
    for s, p, o in triples:
        if not ((s in valid or s in test) and p == 'locatedIn' and o in region_names):
            s1_triples |= {(s, p, o)}

    write_triples_to_file('s1/triples_s1.tsv', s1_triples)

    if not os.path.exists('s2'):
        os.makedirs('s2')

    s2_triples = set()
    for s, p, o in triples:
        if not ((s in valid or s in test) and p == 'locatedIn' and o in region_names):
            if not ((s in valid or s in test) and p == 'locatedIn' and o in subregion_names):
                s2_triples |= {(s, p, o)}

    write_triples_to_file('s2/triples_s2.tsv', s2_triples)

    if not os.path.exists('s3'):
        os.makedirs('s3')

    def has_neighbor_in_test(country_name):
        for _s, _p, _o in triples:
            if _s == country_name and _p == 'neighborOf' and (_o in valid or _o in test):
                return True
        return False

    s3_triples = set()
    for s, p, o in triples:
        if not ((s in valid or s in test) and p == 'locatedIn' and o in region_names):
            if not ((s in valid or s in test) and p == 'locatedIn' and o in subregion_names):
                if not (has_neighbor_in_test(s) and p == 'locatedIn' and o in region_names):
                    s3_triples |= {(s, p, o)}

    write_triples_to_file('s3/triples_s2.tsv', s3_triples)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
