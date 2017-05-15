#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

dataset_name = 'FB15k_clean'

rdm = np.random.RandomState(2342423)
files = ['wn18/wordnet-mlj12-train.txt', 'wn18/wordnet-mlj12-valid.txt', 'wn18/wordnet-mlj12-test.txt']

data = []
for p in files:
    with open(p) as f:
        data = f.readlines() + data

e_set = set()
test_cases = {}
rel_to_tuple = {}
for p in files:
    test_cases[p] = []


for p in files:
    with open(p) as f:
        for i, line in enumerate(f):
            e1, rel, e2 = line.split('\t')
            e1, e2, rel = e1.strip(), e2.strip(), rel.strip()

            e_set.add(e1)
            e_set.add(e2)

            if rel not in rel_to_tuple:
                rel_to_tuple[rel] = set()

            rel_to_tuple[rel].add((e1, e2))
            test_cases[p].append([e1, rel, e2])


def check_for_reversible_relations(rel_to_tuple):
    rel2reversal_rel = {}
    for i, rel1 in enumerate(rel_to_tuple):
        print('Processed {0} relations...'.format(i))
        for rel2 in rel_to_tuple:
            tuples2 = rel_to_tuple[rel2]
            tuples1 = rel_to_tuple[rel1]
            # check if the entire set of (e1, e2) is contained in the set of the
            # other relation, but in a reversed manner
            # that is ALL (e1, e2) -> (e2, e1) for rel 1 are contained in set entity tuple set of rel2 (and vice versa)
            # if this is true for ALL entities, that is the sets completely overlap, then add a rule that
            # (e1, rel1, e2) == (e2, rel2, e1)
            left = all([(e2, e1) in tuples2 for (e1, e2) in tuples1])
            right = all([(e1, e2) in tuples1 for (e2, e1) in tuples2])
            if left or right:
                rel2reversal_rel[rel1] = rel2
                rel2reversal_rel[rel2] = rel1
    return rel2reversal_rel

rel2reversal_rel = check_for_reversible_relations(rel_to_tuple)
