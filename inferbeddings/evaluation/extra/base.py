# -*- coding: utf-8 -*-

import abc
import numpy as np

from sklearn import metrics
from sklearn.preprocessing import normalize

import inferbeddings.evaluation.extra.davis as davis

import math
import logging


class RankingEvaluationMetric(metaclass=abc.ABCMeta):
    """
    Abstract class inherited by all Evaluation Metrics.
    """
    def __init__(self, pos_label=1, normalize_scores=True):
        self.pos_label = pos_label
        self.normalize_scores = normalize_scores

    def _preprocess_scores(self, scores):
        """
        Normalizes a vector of scores.
        :param scores: Vector of scores.
        :return: Normalized scores.
        """
        preprocessed_scores = scores
        if self.normalize_scores is True:
            preprocessed_scores = normalize(preprocessed_scores.reshape(-1, 1), axis=0).ravel()
        return preprocessed_scores

    @abc.abstractmethod
    def __call__(self, y, scores):
        while False:
            yield None

    @property
    @abc.abstractmethod
    def name(self):
        while False:
            yield None


class AUCPRDavis(RankingEvaluationMetric):
    """
    Area Under the Precision-Recall Curve (AUC-PR), calculated using the procedure described in [1].

    [1] J Davis et al. - The Relationship Between Precision-Recall and ROC Curves - ICML 2006
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, y, scores):
        scores = self._preprocess_scores(scores)
        n, n_pos = len(scores), np.sum(y == self.pos_label)

        metric = davis.AUC(n_pos, n - n_pos)
        order = np.argsort(scores)[::-1]
        ordered_y = y[order]

        _tp = np.sum(y == self.pos_label)

        pn_points = []
        for i in reversed(range(y.shape[0])):
            _y = ordered_y[:i + 1]
            n = _y.shape[0]

            _tp -= 1 if (i + 1) < ordered_y.shape[0] and ordered_y[i + 1] == self.pos_label else 0
            fp = n - _tp

            point = davis.PNPoint(_tp, fp)
            pn_points += [point]

        metric.set_pn_points(pn_points)
        metric.interpolate()
        ans = metric.calculate_auc_pr()
        return ans

    @property
    def name(self):
        return 'AUC-PR (Davis)'


class AUCPRSciKit(RankingEvaluationMetric):
    """
    Area Under the Precision-Recall Curve (AUC-PR), calculated using scikit-learn.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, y, scores):
        scores = self._preprocess_scores(scores)
        precision, recall, thresholds = metrics.precision_recall_curve(y, scores, pos_label=self.pos_label)
        ans = metrics.auc(recall, precision)
        return ans

    @property
    def name(self):
        return 'AUC-PR (scikit-learn)'


class AUCROCDavis(RankingEvaluationMetric):
    """
    Area Under the ROC Curve (AUC-ROC), calculated using the procedure described in [1].

    [1] J Davis et al. - The Relationship Between Precision-Recall and ROC Curves - ICML 2006
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, y, scores):
        scores = self._preprocess_scores(scores)
        n, n_pos = len(scores), np.sum(y == self.pos_label)

        metric = davis.AUC(n_pos, n - n_pos)
        order = np.argsort(scores)[::-1]
        ordered_y = y[order]

        _tp = np.sum(y == self.pos_label)

        pn_points = []
        for i in reversed(range(y.shape[0])):
            _y = ordered_y[:i + 1]
            n = _y.shape[0]

            _tp -= 1 if (i + 1) < ordered_y.shape[0] and ordered_y[i + 1] == self.pos_label else 0
            fp = n - _tp

            point = davis.PNPoint(_tp, fp)
            pn_points += [point]

        metric.set_pn_points(pn_points)
        metric.interpolate()
        ans = metric.calculate_auc_roc()
        return ans

    @property
    def name(self):
        return 'AUC-ROC (Davis)'


class AUCROCSciKit(RankingEvaluationMetric):
    """
    Area Under the Precision-Recall Curve (AUC-PR), calculated using scikit-learn.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, y, scores):
        scores = self._preprocess_scores(scores)
        ans = metrics.roc_auc_score((y == self.pos_label).astype(int), scores)
        return ans

    @property
    def name(self):
        return 'AUC-ROC (scikit-learn)'


AUCPR = AUCPRSciKit
AUCROC = AUCROCSciKit


class HitsAtK(RankingEvaluationMetric):
    """
    Hits@K: Number of correct elements retrieved among the K elements with the
    highest score.
    """
    def __init__(self, k=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def __call__(self, y, scores):
        scores = self._preprocess_scores(scores)

        k = len(y) if self.k is None else self.k

        return (y[np.argsort(scores)[::-1]][:k] == self.pos_label).sum()

    @property
    def name(self):
        return 'Hits@%d' % self.k


class PrecisionAtK(RankingEvaluationMetric):
    """
    Precision@K [1]: Fraction of relevant elements retrieved among the K elements with the highest score.

    [1] T Y Liu - Learning to Rank for Information Retrieval - Springer 2011
    """
    def __init__(self, k=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def __call__(self, y, scores):
        scores = self._preprocess_scores(scores)
        order = np.argsort(scores)[::-1]
        n_pos = np.sum(y == self.pos_label)

        k = len(y) if self.k is None else self.k

        n_relevant = np.sum(y[order[:k]] == self.pos_label)
        return float(n_relevant) / min(n_pos, self.k)

    @property
    def name(self):
        return 'Precision' + (('@%s' % self.k) if self.k is not None else '')


class AveragePrecision(RankingEvaluationMetric):
    """
    Average Precision [1]

    [1] T Y Liu - Learning to Rank for Information Retrieval - Springer 2011
    """
    def __init__(self, k=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def __call__(self, y, scores):
        scores = self._preprocess_scores(scores)
        order = np.argsort(scores)[::-1]

        k = len(y) if self.k is None else self.k
        _y, _order = y[:k], order[:k]
        n, ord_y = _y.shape[0], _y[_order]

        num, n_pos = .0, 0
        for i in range(n):
            n_pos += 1 if ord_y[i] == self.pos_label else 0
            num += (n_pos / (i + 1)) if ord_y[i] == self.pos_label else .0

        return num / n_pos

    @property
    def name(self):
        return 'Average Precision' + (('@%s' % self.k) if self.k is not None else '')


class DCG(RankingEvaluationMetric):
    """
    Discounted Cumulative Gain [1]

    [1] T Y Liu - Learning to Rank for Information Retrieval - Springer 2011
    """
    def __init__(self,
                 k=None,
                 G=lambda x: 2 ** x - 1,
                 eta=lambda x: 1 / math.log2(x + 1),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.G = G
        self.eta = eta

    def __call__(self, y, scores):
        scores = self._preprocess_scores(scores)
        order = np.argsort(scores)[::-1]
        ord_y, dcg = y[order], .0

        k = len(y) if self.k is None else self.k

        for i in range(k):
            dcg += self.G(ord_y[i] == self.pos_label) * self.eta(i + 1)
        return dcg

    @property
    def name(self):
        return 'DCG' + (('@%s' % self.k) if self.k is not None else '')


class NDCG(RankingEvaluationMetric):
    """
    Normalized Discounted Cumulative Gain [1]

    [1] T Y Liu - Learning to Rank for Information Retrieval - Springer 2011
    """
    def __init__(self,
                 k=None,
                 G=lambda x: 2 ** x - 1,
                 eta=lambda x: 1 / math.log2(x + 1),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.dcg = DCG(k=self.k, G=G, eta=eta, *args, **kwargs)

    def __call__(self, y, scores):
        dcg_score = self.dcg(y, scores)
        normalization_term = self.dcg(y, y)
        return dcg_score / normalization_term

    @property
    def name(self):
        return 'NDCG' + (('@%s' % self.k) if self.k is not None else '')
