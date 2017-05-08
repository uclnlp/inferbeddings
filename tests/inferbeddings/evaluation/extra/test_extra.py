# -*- coding: utf-8 -*-

import numpy as np

import inferbeddings.evaluation.extra.base as metrics
import inferbeddings.evaluation.extra.davis as davis

import math

import logging
import time
import unittest


class TestMetrics(unittest.TestCase):

    def setUp(self):
        pass

    def aucpr_davis(self, y, scores, norm=False):
        return metrics.AUCPRDavis(normalize_scores=norm)(y, scores)

    def aucpr_sklearn(self, y, scores, norm=False):
        return metrics.AUCPRSciKit(normalize_scores=norm)(y, scores)

    def aucroc_davis(self, y, scores, norm=False):
        return metrics.AUCROCDavis(normalize_scores=norm)(y, scores)

    def aucroc_sklearn(self, y, scores, norm=False):
        return metrics.AUCROCSciKit(normalize_scores=norm)(y, scores)

    def ndcg(self, y, scores, norm=False):
        return metrics.NDCG(normalize_scores=norm)(y, scores)

    def test_ranking_score(self):
        n = 8192

        time_davis, time_sklearn = .0, .0

        random_state = np.random.RandomState(0)
        for _ in range(2 ** 3):
            y = random_state.randint(2, size=n)
            scores = random_state.rand(n)

            t_zero = time.time()
            value_davis = self.aucpr_davis(y, scores)
            t_one = time.time()
            value_sklearn = self.aucpr_sklearn(y, scores)
            t_two = time.time()

            time_davis += t_one - t_zero
            time_sklearn += t_two - t_one

            self.assertAlmostEqual(value_davis, value_sklearn, places=6)

            value_davis = self.aucroc_davis(y, scores)
            value_sklearn = self.aucroc_sklearn(y, scores)
            self.assertAlmostEqual(value_davis, value_sklearn, places=6)

    def test_auc(self):
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
        self.assertAlmostEqual(metric.calculate_auc_pr(), 0.919642857143)

    def test_ndcg(self):
        y = np.array([1, 1, 1, 1, 1, 0, 0, 0])
        scores = np.array([1., .95, .8, .7, .6, .5, .4, .3])

        value = self.ndcg(y, scores)
        self.assertAlmostEqual(value, 1.0)

        y = np.array([0, 1, 1, 0, 1, 0, 0, 0])
        scores = np.array([1., .95, .8, .7, .6, .5, .4, .3])

        value = self.ndcg(y, scores)

    def test_dcg(self):
        y = np.array([1, 1, 1, 1, 0, 0])
        scores = np.array([1., .9, .8, .7, .6, .5])

        G = lambda x: x
        eta = lambda x: 1 if x < 1.5 else 1 / math.log2(x)

        dcg = metrics.DCG(normalize_scores=False, G=G, eta=eta, k=len(y))
        value = dcg(y, scores)

        # 1/1 + 1/log2(2) + 1/log2(3) + 1/log2(4) = 3.131
        self.assertAlmostEqual(value, 3.131, places=2)

        G = lambda x: 2 ** x - 1
        eta = lambda x: 1 / math.log2(x + 1)

        dcg = metrics.DCG(normalize_scores=False, G=G, eta=eta, k=len(y))
        value = dcg(y, scores)

        # (2**1 - 1)/log2(1 + 1) + (2**1 - 1)/log2(2 + 1) +
        # (2**1 - 1)/log2(3 + 1) + (2**1 - 1)/log2(4 + 1) = 2.562
        self.assertAlmostEqual(value, 2.562, places=2)

        y = np.array([1, 1, 0, 1, 0, 0])
        scores = np.array([1., .9, .8, .7, .6, .5])

        value_lower = dcg(y, scores)

        self.assertAlmostEqual(value_lower, 2.062, places=2)
        self.assertTrue(value_lower < value)

        y = np.array([0, 1, 0, 1, 0, 1])
        scores = np.array([.5, .9, .8, .7, .6, 1.])

        value_lower_2 = dcg(y, scores)

        self.assertAlmostEqual(value_lower_2, 2.062, places=2)

    def test_ap(self):
        y = np.array([1, 1, 0, 1, 0, 1, 0, 0, 0, 1])
        scores = np.array([1., .9, .8, .7, .6, .5, .4, .3, .2, .1])

        ap = metrics.AveragePrecision(normalize_scores=False)
        value = ap(y, scores)

        self.assertAlmostEqual(value, 0.78333, places=3)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
