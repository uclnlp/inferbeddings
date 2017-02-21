#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns

df = pd.DataFrame.from_csv('./results/results.tsv', sep='\t', index_col=None)


def select(df, metric="HITS@10", model="TransE"):
    if metric is not None:
        df = df[(df.Metric == metric)]
    if model is not None:
        df = df[(df.Model == model)]
    return df


df = select(df, metric=None, model=None)

g = sns.FacetGrid(df, col="Model", row="Metric", hue="Method", sharey=False,
                  size=4, legend_out=False, despine=False, margin_titles=True)

g.map(sns.regplot, "Fraction", "Result", fit_reg=None)

g.add_legend()

g.savefig("./results/results.pdf")

