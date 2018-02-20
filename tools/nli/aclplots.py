#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob

import pandas as pd
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
    title_fontsize = 20
    fontsize = 20
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

    logging.info('Producing plots..')

    sns.set_style("white")
    sns.set_style("ticks")

    to_str = {
        'dam': 'DAM',
        'cbilstm': 'cBiLSTM',
        'esim': 'ESIM'
    }

    for model_name in ['dam', 'cbilstm', 'esim']:
        data = {'x': [], 'y': [], 'class': []}

        for i, ii in enumerate([100, 500, 1000, 2000, 3000, 4000, 5000, 'full']):
            if isinstance(ii, int) and ii <= 2000:
                accuracies = [get_accuracy(s) for s in results['/k_v12/v1/X_{}_{}_test'.format(model_name, ii)]]
                for lmbda_idx, accuracy in enumerate(accuracies):
                    if lmbda_idx <= 4:
                        dataset_name = '$\mathcal{A}_{\mathrm{' + to_str[model_name] + '}}^{' + str(ii) + '}$'
                        data['class'] += [dataset_name]

                        lmbdas = ["$0.0$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$1.0$"]
                        data['x'] += [lmbdas[lmbda_idx]]
                        data['y'] += [accuracy]

        df = pd.DataFrame(data)

        # Optimal: size=3, aspect=2
        for size in [3]:
            for aspect in [2]:
                logging.info('Size: {}, Aspect: {}'.format(size, aspect))

                palette = None
                g = sns.factorplot(x="x", y="y", hue="class", palette=palette, data=df,
                                   linestyles=[":", "-.", "--", "-"], markers=['o', 'v', "<", ">"],
                                   legend=False, size=size, aspect=aspect)

                start = 0.0
                if model_name == 'dam':
                    start = 0.4
                elif model_name == 'esim':
                    start = 0.5
                elif model_name == 'cbilstm':
                    start = 0.7
                # g.set(ylim=(None, 1.0))

                g.fig.get_axes()[0].legend(loc='lower right', title='Dataset', fontsize=labelsize)

                plt.grid()
                plt.title('Accuracy on $\mathcal{A}_{\mathrm{' + to_str[model_name] + '}}^{k}$ for varying $\lambda$',
                          fontsize=title_fontsize)
                plt.xlabel('Regularisation Parameter $\lambda$', fontsize=fontsize)
                plt.ylabel('Accuracy', fontsize=fontsize)

                g.savefig('acl/plots/accuracy_adversarial_{}_{}_{}.pdf'.format(model_name, size, aspect))

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
                data['y'] += [perc * 100]

    df = pd.DataFrame(data)

    # Optimal: size=3, aspect=2
    for size in [3]:
        for aspect in [2]:
            logging.info('Size: {}, Aspect: {}'.format(size, aspect))

            palette = None
            g = sns.factorplot(x="x", y="y", hue="class", palette=palette, data=df,
                               linestyles=[":", "-.", "--", "-"], markers=['o', 'v', "<", ">"],
                               legend=False, size=size, aspect=aspect)

            g.fig.get_axes()[0].legend(loc='upper right', title='Rules', fontsize=labelsize)

            plt.grid()
            plt.title('Number of violations (%) for varying values of $\lambda$',
                      fontsize=title_fontsize)
            plt.xlabel('Regularisation Parameter $\lambda$', fontsize=fontsize)
            plt.ylabel('Violations (%)', fontsize=fontsize)

            g.savefig('acl/plots/test_violations_{}_{}.pdf'.format(size, aspect))

    data = {'x': [], 'y': [], 'class': []}

    for data_name in ['', 'd', 't']:
        percs = [get_accuracy(s) for s in results['/k_v12/X{}'.format(data_name)]]
        lmbdas = ["$0.0$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$1.0$"]

        name = None
        if data_name == '':
            name = 'Train'
        if data_name == 'd':
            name = 'Dev'
        if data_name == 't':
            name = 'Test'

        for perc_idx, perc in enumerate(percs):
            lmbda_idx = perc_idx

            if lmbda_idx <= 4:
                data['class'] += [name]
                data['x'] += [lmbdas[lmbda_idx]]
                data['y'] += [perc * 100]

    df = pd.DataFrame(data)

    # Optimal: size=3, aspect=2
    for size in [3]:
        for aspect in [2]:
            logging.info('Size: {}, Aspect: {}'.format(size, aspect))

            palette = sns.color_palette("cubehelix", 3)

            g = sns.factorplot(x="x", y="y", hue="class", palette=palette, data=df,
                               linestyles=[":", "--", "-"], markers=['o', "<", ">"],
                               legend=False, size=size, aspect=aspect)

            g.fig.get_axes()[0].legend(loc='upper right', title='SNLI', fontsize=labelsize)

            plt.grid()
            plt.title('DAM Accuracy on SNLI',
                      fontsize=title_fontsize)
            plt.xlabel('Regularisation Parameter $\lambda$', fontsize=fontsize)
            plt.ylabel('Accuracy (%)', fontsize=fontsize)

            g.savefig('acl/plots/accuracy_dam_{}_{}.pdf'.format(size, aspect))

    data = {'x': [], 'y': [], 'class': []}

    for data_name in ['', 'd', 't']:
        percs = [get_accuracy(s) for s in results['/k_v12c/X{}.cbilstm'.format(data_name)]]
        lmbdas = ["$0.0$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$1.0$"]

        name = None
        if data_name == '':
            name = 'Train'
        if data_name == 'd':
            name = 'Valid.'
        if data_name == 't':
            name = 'Test'

        for perc_idx, perc in enumerate(percs):
            lmbda_idx = perc_idx

            if lmbda_idx <= 4:
                data['class'] += [name]
                data['x'] += [lmbdas[lmbda_idx]]
                data['y'] += [perc * 100]

    df = pd.DataFrame(data)

    # Optimal: size=3, aspect=2
    for size in [3]:
        for aspect in [2]:
            logging.info('Size: {}, Aspect: {}'.format(size, aspect))

            palette = sns.color_palette("cubehelix", 3)

            g = sns.factorplot(x="x", y="y", hue="class", palette=palette, data=df,
                               linestyles=[":", "--", "-"], markers=['o', "<", ">"],
                               legend=False, size=size, aspect=aspect)

            g.fig.get_axes()[0].legend(loc='upper right', title='SNLI', fontsize=labelsize)

            plt.grid()
            plt.title('cBiLSTM Accuracy on SNLI',
                      fontsize=title_fontsize)
            plt.xlabel('Regularisation Parameter $\lambda$', fontsize=fontsize)
            plt.ylabel('Accuracy (%)', fontsize=fontsize)

            g.savefig('acl/plots/accuracy_cbilstm_{}_{}.pdf'.format(size, aspect))

    logging.info('Producing tables..')

    # This row will contain column titles
    row_0 = ['\\diagbox{\\bf Model}{\\bf Dataset}']

    # Results for regularised and unregularised DAM
    row_1 = ['{\\bf DAM}$^{\mathcal{AR}}$']
    row_2 = ['{\\bf DAM}']

    # Results for regularised and unregularised cBiLSTM
    row_3 = ['{\\bf cBiLSTM}$^{\mathcal{AR}}$']
    row_4 = ['{\\bf cBiLSTM}']

    data_sizes = [100, 500, 1000, 2000]
    data_models = ['dam', 'esim', 'cbilstm']

    for data_model in data_models:
        for data_size in data_sizes:

            data_model_name = None
            if data_model == 'dam':
                data_model_name = 'DAM'
            if data_model == 'esim':
                data_model_name = 'ESIM'
            elif data_model == 'cbilstm':
                data_model_name = 'cBiLSTM'

            dataset_name = '$\\aset{' + data_model_name + '}{' + str(data_size) + '}$'

            dam_dev_accs = [get_accuracy(s) for s in results['/k_v12/v1/X_{}_{}_dev'.format(data_model, data_size)]]
            dam_test_accs = [get_accuracy(s) for s in results['/k_v12/v1/X_{}_{}_test'.format(data_model, data_size)]]

            dam_best_dev_acc_idx = dam_dev_accs.index(max(dam_dev_accs))
            dam_test_acc = dam_test_accs[dam_best_dev_acc_idx]

            cbilstm_dev_accs = [get_accuracy(s) for s in results['/k_v12c/v1/X_{}_{}_dev.cbilstm'.format(data_model, data_size)]]
            cbilstm_test_accs = [get_accuracy(s) for s in results['/k_v12c/v1/X_{}_{}_test.cbilstm'.format(data_model, data_size)]]

            cbilstm_best_dev_acc_idx = cbilstm_dev_accs.index(max(cbilstm_dev_accs))
            cbilstm_test_acc = cbilstm_test_accs[cbilstm_best_dev_acc_idx]

            row_0 += [dataset_name]

            row_1 += [dam_test_acc]
            row_2 += [dam_test_accs[0]]

            row_3 += [cbilstm_test_acc]
            row_4 += [cbilstm_test_accs[0]]

    table_str = """
\\begin{tabular}{""" + ''.join(['R{3cm}'] + (['|C{1.5cm}C{1.5cm}C{1.5cm}C{1.5cm}'] * (len(data_models)))) + """}
\\toprule
"""

    table_str += ' & '.join(row_0) + " \\\\ \n\\midrule\n"

    for i, row in enumerate([row_1, row_2]):
        def p(s):
            return s if i > 0 else '{\\bf ' + s + '}'
        table_str += ' & '.join([p(str(e)) for e in row]) + " \\\\ \n"

    table_str += "\n\\midrule\n"

    for i, row in enumerate([row_3, row_4]):
        def p(s):
            return s if i > 0 else '{\\bf ' + s + '}'
        table_str += ' & '.join([p(str(e)) for e in row]) + " \\\\ \n"

    table_str += """
\\bottomrule
\\end{tabular}
"""

    with open('acl/tables/accuracy.tex', 'w') as f:
        f.write(table_str)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
