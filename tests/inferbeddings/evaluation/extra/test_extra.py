# -*- coding: utf-8 -*-

import pytest

import numpy as np

import inferbeddings.evaluation.extra.base as metrics
import inferbeddings.evaluation.extra.davis as davis

import math

import time


def aucpr_davis(y, scores, norm=False):
    return metrics.AUCPRDavis(normalize_scores=norm)(y, scores)


def aucpr_sklearn(y, scores, norm=False):
    return metrics.AUCPRSciKit(normalize_scores=norm)(y, scores)


def aucroc_davis(y, scores, norm=False):
    return metrics.AUCROCDavis(normalize_scores=norm)(y, scores)


def aucroc_sklearn(y, scores, norm=False):
    return metrics.AUCROCSciKit(normalize_scores=norm)(y, scores)


def ndcg(y, scores, norm=False):
    return metrics.NDCG(normalize_scores=norm)(y, scores)


@pytest.mark.light
def test_ranking_score():
    n = 8192

    time_davis, time_sklearn = .0, .0

    random_state = np.random.RandomState(0)
    for _ in range(2 ** 3):
        y = random_state.randint(2, size=n)
        scores = random_state.rand(n)

        t_zero = time.time()
        value_davis = aucpr_davis(y, scores)
        t_one = time.time()
        value_sklearn = aucpr_sklearn(y, scores)
        t_two = time.time()

        time_davis += t_one - t_zero
        time_sklearn += t_two - t_one

        np.testing.assert_allclose(value_davis, value_sklearn, rtol=1e-4)

        value_davis = aucroc_davis(y, scores)
        value_sklearn = aucroc_sklearn(y, scores)
        np.testing.assert_allclose(value_davis, value_sklearn, rtol=1e-4)


@pytest.mark.light
def test_auc():
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0])
    scores = np.array([1., .2, .8, .7, .6, .5, .4, .3])

    metric = davis.AUC(np.sum([y == 1]), np.sum([y == 0]))

    order = np.argsort(scores)[::-1]
    for i in range(y.shape[0]):
        _y = y[order][:i + 1]
        tp, fp = np.sum([_y == 1]), np.sum([_y == 0])
        point = davis.PNPoint(tp, fp)
        metric.add_pn_point(point)

    metric.interpolate()
    np.testing.assert_allclose(metric.calculate_auc_pr(), 0.919642857143)


@pytest.mark.light
def test_ndcg():
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0])
    scores = np.array([1., .95, .8, .7, .6, .5, .4, .3])

    value = ndcg(y, scores)
    np.testing.assert_allclose(value, 1.0)

    y = np.array([0, 1, 1, 0, 1, 0, 0, 0])
    scores = np.array([1., .95, .8, .7, .6, .5, .4, .3])

    value = ndcg(y, scores)


@pytest.mark.light
def test_dcg():
    y = np.array([1, 1, 1, 1, 0, 0])
    scores = np.array([1., .9, .8, .7, .6, .5])

    G = lambda x: x
    eta = lambda x: 1 if x < 1.5 else 1 / math.log2(x)

    dcg = metrics.DCG(normalize_scores=False, G=G, eta=eta, k=len(y))
    value = dcg(y, scores)

    # 1/1 + 1/log2(2) + 1/log2(3) + 1/log2(4) = 3.131
    np.testing.assert_allclose(value, 3.131, rtol=1e-2)

    G = lambda x: 2 ** x - 1
    eta = lambda x: 1 / math.log2(x + 1)

    dcg = metrics.DCG(normalize_scores=False, G=G, eta=eta, k=len(y))
    value = dcg(y, scores)

    # (2**1 - 1)/log2(1 + 1) + (2**1 - 1)/log2(2 + 1) +
    # (2**1 - 1)/log2(3 + 1) + (2**1 - 1)/log2(4 + 1) = 2.562
    np.testing.assert_allclose(value, 2.562, rtol=1e-2)

    y = np.array([1, 1, 0, 1, 0, 0])
    scores = np.array([1., .9, .8, .7, .6, .5])

    value_lower = dcg(y, scores)

    np.testing.assert_allclose(value_lower, 2.062, rtol=1e-2)
    assert value_lower < value

    y = np.array([0, 1, 0, 1, 0, 1])
    scores = np.array([.5, .9, .8, .7, .6, 1.])

    value_lower_2 = dcg(y, scores)

    np.testing.assert_allclose(value_lower_2, 2.062, rtol=1e-2)


@pytest.mark.light
def test_ap():
    y = np.array([1, 1, 0, 1, 0, 1, 0, 0, 0, 1])
    scores = np.array([1., .9, .8, .7, .6, .5, .4, .3, .2, .1])

    ap = metrics.AveragePrecision(normalize_scores=False)
    value = ap(y, scores)

    np.testing.assert_allclose(value, 0.78333, rtol=1e-2)


if __name__ == '__main__':
    pytest.main([__file__])
