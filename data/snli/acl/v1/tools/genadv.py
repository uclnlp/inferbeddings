#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse
import requests

import pickle
import atexit
import operator
import copy

import numpy as np

import gzip
import json

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def contradiction_loss_dam(sentence1, sentence2):
    p1 = call_dam(sentence1, sentence2)['contradiction']
    p2 = call_dam(sentence2, sentence1)['contradiction']
    p1, p2 = float(p1), float(p2)
    return max(p1 - p2, 0), max(p2 - p1, 0)


def contradiction_loss_esim(sentence1, sentence2):
    p1 = call_esim(sentence1, sentence2)['contradiction']
    p2 = call_esim(sentence2, sentence1)['contradiction']
    p1, p2 = float(p1), float(p2)
    return max(p1 - p2, 0), max(p2 - p1, 0)


def contradiction_loss_cbilstm(sentence1, sentence2):
    p1 = call_cbilstm(sentence1, sentence2)['contradiction']
    p2 = call_cbilstm(sentence2, sentence1)['contradiction']
    p1, p2 = float(p1), float(p2)
    return max(p1 - p2, 0), max(p2 - p1, 0)


def persist(path):
    def decorator(fun):
        cache = {}
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                cache = pickle.load(f)

        def write():
            with open(path, 'wb') as f:
                pickle.dump(cache, f)
        atexit.register(lambda: write())

        def new_f(*args):
            if tuple(args) not in cache:
                cache[tuple(args)] = fun(*args)
            return cache[args]
        return new_f
    return decorator


@persist('dam_cache.p')
def call_dam(sentence1, sentence2, url='http://127.0.0.1:8889/v1/nli'):
    data = {'sentence1': sentence1, 'sentence2': sentence2}
    ans = requests.post(url, data=data)
    ans_json = ans.json()
    return ans_json


@persist('esim_cache.p')
def call_esim(sentence1, sentence2, url='http://127.0.0.1:9001/v1/nli'):
    data = {'sentence1': sentence1, 'sentence2': sentence2}
    ans = requests.post(url, data=data)
    ans_json = ans.json()
    return ans_json


@persist('cbilstm_cache.p')
def call_cbilstm(sentence1, sentence2, url='http://127.0.0.1:9002/v1/nli'):
    data = {'sentence1': sentence1, 'sentence2': sentence2}
    ans = requests.post(url, data=data)
    ans_json = ans.json()
    return ans_json


def invert(_obj):
    i_obj = copy.deepcopy(_obj)

    # Switch sentence1 with sentence2
    for i, j in [(1, 2), (2, 1)]:
        for suf in ['', '_binary_parse', '_parse']:
            i_obj['sentence{}{}'.format(i, suf)] = _obj['sentence{}{}'.format(j, suf)]

    # Heuristically change the gold_label
    gold_label = _obj['gold_label']
    assert gold_label in {'entailment', 'contradiction', 'neutral', '-'}

    if gold_label == 'entailment':
        i_obj['gold_label'] = 'neutral'
    if gold_label == 'neutral':
        i_obj['gold_label'] = 'neutral'
    if gold_label == 'contradiction':
        i_obj['gold_label'] = 'contradiction'

    return i_obj


def main(argv):
    def fmt(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('SNLI Adversarial Candidate Generator', formatter_class=fmt)

    argparser.add_argument('--path', '-p', action='store', type=str, default='snli_1.0_train.jsonl.gz')
    argparser.add_argument('--seed', '-s', action='store', type=int, default=0)
    argparser.add_argument('--fraction', '-f', action='store', type=float, default=None)
    argparser.add_argument('--nb-instances', '-n', action='store', type=int, default=None)

    args = argparser.parse_args(argv)

    path = args.path
    seed = args.seed
    fraction = args.fraction
    nb_instances = args.nb_instances

    obj_lst = []
    with gzip.open(path, 'rb') as f:
        for line in f:
            dl = line.decode('utf-8')
            obj_lst += [json.loads(dl)]

    if fraction is not None:
        rs = np.random.RandomState(seed)
        nb_obj = len(obj_lst)
        # Round to the closest integer
        nb_samples = int(round(nb_obj * fraction))

        sample_idxs = rs.choice(nb_obj, nb_samples, replace=False)
        sample_obj_lst = [obj_lst[i] for i in sample_idxs]
    else:
        sample_obj_lst = obj_lst

    obj_c_loss_dam_pairs = []
    obj_c_loss_esim_pairs = []
    obj_c_loss_cbilstm_pairs = []

    obj_c_loss_pairs = []

    for obj in sample_obj_lst:
        s1, s2 = obj['sentence1'], obj['sentence2']

        dam_c1,  dam_c2 = contradiction_loss_dam(s1, s2)
        esim_c1, esim_c2 = contradiction_loss_esim(s1, s2)
        cbilstm_c1, cbilstm_c2 = contradiction_loss_cbilstm(s1, s2)

        c_loss_value_dam = dam_c1 + dam_c2
        c_loss_value_esim = esim_c1 + esim_c2
        c_loss_value_cbilstm = cbilstm_c1 + cbilstm_c2

        obj_c_loss_dam_pairs += [(obj, c_loss_value_dam)]
        obj_c_loss_esim_pairs += [(obj, c_loss_value_esim)]
        obj_c_loss_cbilstm_pairs += [(obj, c_loss_value_cbilstm)]

        obj_c_loss_pairs += [(obj, c_loss_value_dam + c_loss_value_esim + c_loss_value_cbilstm)]

    sorted_objs_c_loss_pairs = sorted(obj_c_loss_pairs,
                                      key=operator.itemgetter(1),
                                      reverse=True)

    if nb_instances is None:
        nb_instances = len(sorted_objs_c_loss_pairs)

    for obj, c_loss in sorted_objs_c_loss_pairs[:nb_instances]:
        s1, s2 = obj['sentence1'], obj['sentence2']

        dam_c1,  dam_c2 = contradiction_loss_dam(s1, s2)
        esim_c1, esim_c2 = contradiction_loss_esim(s1, s2)
        cbilstm_c1, cbilstm_c2 = contradiction_loss_cbilstm(s1, s2)

        c_obj = copy.deepcopy(obj)
        i_obj = invert(obj)

        c_obj['type'] = 'normal'
        i_obj['type'] = 'inverse'

        c_obj['c_loss_dam'] = dam_c1
        i_obj['c_loss_dam'] = dam_c2

        c_obj['c_loss_esim'] = esim_c1
        i_obj['c_loss_esim'] = esim_c2

        c_obj['c_loss_cbilstm'] = cbilstm_c1
        i_obj['c_loss_cbilstm'] = cbilstm_c2

        c_obj['dam'] = call_dam(s1, s2)
        i_obj['dam'] = call_dam(s2, s1)

        c_obj['esim'] = call_esim(s1, s2)
        i_obj['esim'] = call_esim(s2, s1)

        c_obj['cbilstm'] = call_esim(s1, s2)
        i_obj['cbilstm'] = call_esim(s2, s1)

        print(json.dumps(c_obj))
        print(json.dumps(i_obj))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
