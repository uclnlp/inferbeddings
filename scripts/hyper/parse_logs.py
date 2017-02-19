#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os.path

import re

path = '/Users/riedel/projects/inferbeddings/logs/hyper'

print("Weight\tseed\tsubsample\tAUC_ROC_valid\tAUC_ROC_test")

for file in os.listdir(path):
    # /Users/riedel/projects/inferbeddings/logs/hyper/hyper.adv_batch=500_adv_epochs=100_aggregate=max_builder=point-mass_disc_epochs=50_epochs=3_seed=9_subsample_prob=0.01_weight=10.0.log
    seed = re.search(r'seed=(.*?)_', file).group(1)
    subsample_prob = re.search(r'subsample_prob=(.*?)_', file).group(1)
    weight = re.search(r'weight=(.*?).log', file).group(1)
    results = []
    with open(path + "/" + file, "r") as f:
        for line in f.readlines():
            if line.startswith("INFO:inferbeddings.evaluation.base:"):
                match = re.search(r'AUC-ROC: (\d+\.\d+)', line)
                if match:
                    results.append(match.group(1))
    print("{weight}\t{seed}\t{subsample}\t{auc_valid}\t{auc_test}".format(
        weight=weight, subsample=subsample_prob, seed=seed, auc_valid=results[0], auc_test=results[1]))
