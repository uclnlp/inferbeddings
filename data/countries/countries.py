#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
yes, I am measuring AUC-*ROC* on the locatedIn relation

[12:51] 
but only for locatedIn(c,r) where c is a test country and r a region (not a subregion!)

[12:51] 
this is how I understood the experimental setup in Nickel 2015

-- @rockt on Slack :)
"""


import sys
import os

from typing import NamedTuple, List
import json

from tqdm import tqdm
from sklearn.model_selection import train_test_split

import logging
logger = logging.getLogger(os.path.basename(sys.argv[0]))


def norm(name):
    return name.replace(' ', '_').replace("'", '').replace('(', '').replace(')', '').lower()


def write_to_file(path, instances):
    with open(path, 'w') as f:
        for instance in instances:
            f.write('{}\n'.format(instance))


def write_tuples_to_file(path, tuples):
    with open(path, 'w') as f:
        for t in tuples:
            if len(t) == 4:
                s, p, o, i = t
                t = (s, p, o, str(i))
            f.write('{}\n'.format("\t".join(t)))


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
    consistent_set, seed = None, 0

    logger.info('Looking for a train/dev/test split such that for each country\n'
                'in the test set there is at least one neighbor in the training set, as in [1] ..\n\n'
                '[1] https://cbmm.mit.edu/sites/default/files/publications/holographic-embeddings.pdf')

    for seed, _ in tqdm(enumerate(iter(lambda: consistent_set is not None, True), start=180000)):
        logger.debug('Trying seed {} ..'.format(seed))
        train, valid_test = train_test_split(country_names_lst, train_size=0.8, random_state=seed)
        valid, test = train_test_split(valid_test, train_size=0.5, random_state=seed)
        if is_consistent(train, test):
            consistent_set = (train, valid, test)

    train, valid, test = consistent_set

    train_set, valid_set, test_set = set(train), set(valid), set(test)
    assert train_set & valid_set == set()
    assert train_set & test_set == set()
    assert valid_set & test_set == set()
    assert train_set | valid_set | test_set == country_names

    write_to_file('./countries_train.lst', sorted(train))
    write_to_file('./countries_valid.lst', sorted(valid))
    write_to_file('./countries_test.lst', sorted(test))

    if not os.path.exists('s1'):
        os.makedirs('s1')

    s1_triples_train, s1_triples_valid, s1_triples_test = set(), set(), set()
    """
    In the basic setting we only set locatedIn(c, r) to missing for the countries in the test data.
    In this setting, the correct relations can be predicted from patterns of the form:
        locatedIn(c, s) ∧ locatedIn(s, r) ⇒ locatedIn(c, r)
    where s refers to the country’s subregion.
    """
    for s, p, o in triples:
        if s in valid and p == 'locatedIn' and o in region_names:
            # Country c is in the validation set - location(c, r) is in the validation set
            s1_triples_valid |= {(s, p, o, 1)}
            s1_triples_valid |= {(s, p, _o, 0) for _o in region_names if _o != o}
        elif s in test and p == 'locatedIn' and o in region_names:
            # Country c is in the test set - location(c, r) is in the test set
            s1_triples_test |= {(s, p, o, 1)}
            s1_triples_test |= {(s, p, _o, 0) for _o in region_names if _o != o}
        else:
            # All triples not in the validation or test set
            s1_triples_train |= {(s, p, o)}

    write_tuples_to_file('s1/triples.tsv', sorted(triples))
    write_tuples_to_file('s1/s1_train.tsv', sorted(s1_triples_train))
    write_tuples_to_file('s1/s1_valid.tsv', sorted(s1_triples_valid))
    write_tuples_to_file('s1/s1_test.tsv', sorted(s1_triples_test))

    if not os.path.exists('s2'):
        os.makedirs('s2')

    s2_triples_train, s2_triples_valid, s2_triples_test = set(), s1_triples_valid.copy(), s1_triples_test.copy()
    """
    In addition to the instances of S1, we set locatedIn(c, s) to missing for all countries c
    in the test/validation set and all subregions s in the data. In this setting, the correct
    triples can be predicted from:
        neighborOf(c1, c2) ∧ locatedIn(c2, r) ⇒ locatedIn(c1, r)
    This is a harder task than S1, since a country can have multiple neighbors and these can
    be in different regions.
    """
    for s, p, o in s1_triples_train:
        if s in (valid + test) and p == 'locatedIn' and o in subregion_names:
            pass
        else:
            s2_triples_train |= {(s, p, o)}

    write_tuples_to_file('s2/triples.tsv', sorted(triples))
    write_tuples_to_file('s2/s2_train.tsv', sorted(s2_triples_train))
    write_tuples_to_file('s2/s2_valid.tsv', sorted(s2_triples_valid))
    write_tuples_to_file('s2/s2_test.tsv', sorted(s2_triples_test))

    if not os.path.exists('s3'):
        os.makedirs('s3')

    def has_neighbor_in(country_name, test_set):
        for _s, _p, _o in triples:
            if _s == country_name and _p == 'neighborOf' and _o in test_set:
                return True
        return False

    s3_triples_train, s3_triples_valid, s3_triples_test = set(), s2_triples_valid.copy(), s2_triples_test.copy()
    """
    In addition to the instances of S1 and S2 we set all locatedIn(n, r) to missing for all
    neighbors n of all countries in the test/validation set and all regions r in the data.
    In this setting, the correct triples can be predicted from patterns of the form:
        neighborOf(c1, c2) ∧ locatedIn(c2, s) ∧ locatedIn(s, r) ⇒ locatedIn(c1, r)
    This is the most difficult task, as it not only involves the neighborOf relation, but
    also a path of length 3.
    """
    for s, p, o in s2_triples_train:
        if has_neighbor_in(s, valid + test) and p == 'locatedIn' and o in region_names:
            pass
        else:
            s3_triples_train |= {(s, p, o)}

    write_tuples_to_file('s3/triples.tsv', sorted(triples))
    write_tuples_to_file('s3/s3_train.tsv', sorted(s3_triples_train))
    write_tuples_to_file('s3/s3_valid.tsv', sorted(s3_triples_valid))
    write_tuples_to_file('s3/s3_test.tsv', sorted(s3_triples_test))

    assert len(s3_triples_train) < len(s2_triples_train) < len(s1_triples_train)

    for a in s2_triples_train:
        assert a in s1_triples_train

    for a in s3_triples_train:
        assert a in s2_triples_train

    assert s1_triples_valid == s2_triples_valid == s3_triples_valid
    assert s1_triples_test == s2_triples_test == s3_triples_test

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
