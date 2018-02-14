#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

import logging


def get_violations_perc(text, rule_str):
    perc = None
    for line in text.split("\n"):
        if rule_str in line:
            a = line.strip().split(":")
            b = a[3].split("(")[1].replace(")", "")
            perc = float(b)
    return perc


def get_accuracy(text):
    accuracy = None
    for line in text.split("\n"):
        if 'Accuracy' in line:
            accuracy = float(line.split(" ")[1].strip())
    return accuracy


def main(argv):
    title_fontsize = 15
    fontsize = 15
    labelsize = 10

    base_path = '/home/pasquale/ucl/workspace/inferbeddings'
    os.chdir(base_path)

    paths = list(glob.iglob('out_nli/**/*.log'.format(base_path), recursive=True))

    results = dict()

    for path in tqdm(paths):
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        noext = os.path.splitext(basename)[0]

        noext_lst = list(noext)

        number = int(noext_lst[0])
        noext_lst[0] = 'X'
        new_noext = ''.join(noext_lst)

        prefix = dirname.split('ut_nli')[1]
        key = '{}/{}'.format(prefix, new_noext)

        if key not in results:
            results[key] = [None, None, None, None, None, None]

        with open(path, 'r') as f:
            results[key][number - 1] = f.read()

    sns.set_style("white")
    sns.set_style("ticks")

    for model_name in ['dam', 'cbilstm', 'esim']:
        data = {'x': [], 'y': [], 'class': []}

        for i, ii in enumerate([100, 500, 1000, 2000, 3000, 4000, 5000, 'full']):
            if isinstance(ii, int) and ii <= 2000:
                accuracies = [get_accuracy(s) for s in results['/k_v12/v1/X_{}_{}_test'.format(model_name, ii)]]
                for lmbda_idx, accuracy in enumerate(accuracies):
                    if lmbda_idx <= 4:
                        dataset_name = '$\mathcal{A}_{\mathrm{DAM}}^{' + str(ii) + '}$'
                        data['class'] += [dataset_name]

                        lmbdas = ["$0.0$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$1.0$"]
                        data['x'] += [lmbdas[lmbda_idx]]
                        data['y'] += [accuracy]

        df = pd.DataFrame(data)

        # Optimal: size=4, aspect=3
        for size in [3]:
            for aspect in [2]:
                logging.info('Size: {}, Aspect: {}'.format(size, aspect))

                graycolors = sns.mpl_palette('Greys_r', 6)
                g = sns.factorplot(x="x", y="y", hue="class", palette=graycolors, data=df,
                                   linestyles=[":", "-.", "--", "-"], markers=['o', 'v', "<", ">"],
                                   legend=False, size=size, aspect=aspect)

                # g.axes[0][0].legend(loc=1, title='Dataset')
                g.fig.get_axes()[0].legend(loc='lower right', title='Dataset', fontsize=labelsize)

                plt.grid()
                plt.title('Accuracy on adversarial datasets for varying values of $\lambda_{r}$',
                          fontsize=title_fontsize)
                plt.xlabel('Regularisation Parameter $\lambda_{r}$', fontsize=fontsize)
                plt.ylabel('Accuracy', fontsize=fontsize)

                g.savefig('plots/acl/accuracy_adversarial_{}_{}.pdf'.format(size, aspect))

    rule_1 = '(S1 contradicts S2) AND NOT(S2 contradicts S1)'
    rule_2 = '(S1 entailment S2) AND (S2 contradicts S1)'
    rule_3 = '(S1 neutral S2) AND (S2 contradicts S1)'
    rule_4 = '(True) AND NOT(S1 entails S1)'

    str_to_rule = {
        rule_4: '$\mathrm{ent}(X_{1}, X_{1})$',
        rule_1: '$\mathrm{con}(X_{1}, X_{2}) \Rightarrow \mathrm{con}(X_{2}, X_{1})$',
        rule_2: '$\mathrm{ent}(X_{1}, X_{2}) \Rightarrow !\mathrm{con}(X_{2}, X_{1})$',
        rule_3: '$\mathrm{neut}(X_{1}, X_{2}) \Rightarrow !\mathrm{con}(X_{2}, X_{1})$',
    }

    rules = [rule_1, rule_2, rule_3, rule_4]

    data = {'x': [], 'y': [], 'class': []}

    for rule_idx, rule_str in enumerate(rules):
        percs = [get_violations_perc(s, rule_str) for s in results['/k_v12/Xt']]
        for perc_idx, perc in enumerate(percs):
            lmbda_idx = perc_idx
            if lmbda_idx <= 5:
                rule_name = str_to_rule[rule_str]
                data['class'] += [rule_name]
                lmbdas = ["$0.0$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$1.0$"]
                data['x'] += [lmbdas[lmbda_idx]]
                data['y'] += [perc]

    df = pd.DataFrame(data)

    # Optimal: size=4, aspect=3
    for size in [3]:
        for aspect in [2]:
            logging.info('Size: {}, Aspect: {}'.format(size, aspect))

            graycolors = sns.mpl_palette('Greys_r', 6)
            g = sns.factorplot(x="x", y="y", hue="class", palette=graycolors, data=df,
                               linestyles=[":", "-.", "--", "-"], markers=['o', 'v', "<", ">"],
                               legend=False, size=size, aspect=aspect)

            g.axes[0][0].legend(loc='upper right', title='Rules', fontsize=labelsize)

            plt.grid()
            plt.title('Number of violations (%) for varying values of $\lambda_{r}$',
                      fontsize=title_fontsize)
            plt.xlabel('Regularisation Parameter $\lambda_{r}$', fontsize=fontsize)
            plt.ylabel('Violations (%)', fontsize=fontsize)

            g.savefig('plots/acl/test_violations_{}_{}.pdf'.format(size, aspect))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
