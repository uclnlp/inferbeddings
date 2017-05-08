# -*- coding: utf-8 -*-

from functools import total_ordering

import logging


class PRPoint(object):
    """
    Point in the Precision-Recall space.
    """
    def __init__(self, precision, recall):
        self.precision = precision
        self.recall = recall

    @property
    def precision(self):
        """
        Precision = tp / (tp + fp)
        or
        Precision = |{relevant} ∩ {retrieved}| / |{retrieved}|

        :return: precision
        """
        return self._precision

    @precision.setter
    def precision(self, precision):
        assert .0 <= precision <= 1.
        self._precision = precision

    @property
    def recall(self):
        """
        Recall = tp / (tp + fn)
        or
        Recall = |{relevant} ∩ {retrieved}| / |{relevant}|

        :return: recall
        """
        return self._recall

    @recall.setter
    def recall(self, recall):
        assert .0 <= recall <= 1.
        self._recall = recall

    def __str__(self):
        return '(%s, %s)' % (self.precision, self.recall)


class ROCPoint(object):
    """
    Point in the ROC space.
    """
    def __init__(self, true_positive_rate, false_positive_rate):
        self.true_positive_rate = true_positive_rate
        self.false_positive_rate = false_positive_rate

    @property
    def true_positive_rate(self):
        """
        TPR = tp / p = tp / (tp + fp)

        :return: true_positive_rate
        """
        return self._true_positive_rate

    @true_positive_rate.setter
    def true_positive_rate(self, true_positive_rate):
        assert .0 <= true_positive_rate <= 1.
        self._true_positive_rate = true_positive_rate

    @property
    def false_positive_rate(self):
        """
        FPR = fp / n = fp / (fp + tn)

        :return: false_positive_rate
        """
        return self._false_positive_rate

    @false_positive_rate.setter
    def false_positive_rate(self, false_positive_rate):
        assert .0 <= false_positive_rate <= 1.
        self._false_positive_rate = false_positive_rate

    def __str__(self):
        return '(%s, %s)' % (self.true_positive_rate, self.false_positive_rate)


@total_ordering
class PNPoint(object):
    """
    Point in the True Positives/False Positives space.
    """
    def __init__(self, true_positives, false_positives):
        self.true_positives = true_positives
        self.false_positives = false_positives

    @property
    def true_positives(self):
        return self._true_positives

    @true_positives.setter
    def true_positives(self, true_positives):
        assert .0 <= true_positives
        self._true_positives = true_positives

    @property
    def false_positives(self):
        return self._false_positives

    @false_positives.setter
    def false_positives(self, false_positives):
        assert .0 <= false_positives
        self._false_positives = false_positives

    @staticmethod
    def _has_attributes(other):
        return hasattr(other, 'true_positives') and hasattr(other, 'false_positives')

    def __eq__(self, other):
        if not self._has_attributes(other):
            return NotImplemented
        _t = (abs(self.true_positives - other.true_positives) <= 0.001)
        _f = (abs(self.false_positives - other.false_positives) <= 0.001)
        return _t and _f

    def __gt__(self, other):
        if not self._has_attributes(other):
            return NotImplemented
        return (self.true_positives > other.true_positives) or (self.false_positives > other.false_positives)

    def __str__(self):
        return '(%s, %s)' % (self.true_positives, self.false_positives)


class AUC(object):
    """
    Class for interpolating points in the True Positives/False Positives space,
    and calculating the AUC-ROC and AUC-PR.
    """
    def __init__(self, nb_positives, nb_negatives):
        self.nb_positives = nb_positives
        self.nb_negatives = nb_negatives

        self.pn_points = []

    def add_precision_recall_point(self, pr_point):
        """
        Adds a point from the Precision/Recall space.
        :param pr_point: point coordinates.
        """
        # TP =
        # = Recall * P
        # = (TP / (TP + FN)) * P =
        # = (TP / P ) * P
        true_positives = pr_point.recall * self.nb_positives

        # FP =
        # = (TP - Precision * TP) / Precision =
        # = (TP - Precision * TP) * ((TP + FP) / TP) =
        # = (TP + FP) - (TP / (TP + FP)) * TP * ((TP + FP) / TP)
        # = (TP + FP) - TP
        # = FP
        false_positives = (true_positives - pr_point.precision * true_positives) / pr_point.precision

        pn_point = PNPoint(true_positives, false_positives)
        if pn_point not in self.pn_points:
            self.pn_points += [pn_point]
        return

    def add_roc_point(self, roc_point):
        """
        Adds a point from the ROC space.
        :param roc_point: point coordinates.
        """

        # TP =
        # = TPR * P =
        # = (TP / P) * P
        true_positives = roc_point.true_positive_rate * self.nb_positives

        # FP =
        # = FPR * N =
        # = (FP / N) * N
        false_positives = roc_point.false_positive_rate * self.nb_negatives
        pn_point = PNPoint(true_positives, false_positives)
        if pn_point not in self.pn_points:
            self.pn_points += [pn_point]
        return

    def add_pn_point(self, pn_point):
        """
        Adds a point from the True Positives/False Positives space.
        :param pn_point: point coordinate.
        """
        if pn_point not in self.pn_points:
            self.pn_points += [pn_point]
        return

    def set_pn_points(self, pn_points):
        self.pn_points = pn_points
        return

    def interpolate(self):
        sorted_pn_points = sorted(self.pn_points)
        interpolated_points = []

        for idx, A in enumerate(sorted_pn_points[:-1]):
            interpolated_points += [A]
            B = sorted_pn_points[idx + 1]

            true_positives_diff = B.true_positives - A.true_positives
            false_positives_diff = B.false_positives - A.false_positives

            # Number of negative examples needed to equal one positive example
            local_skew = false_positives_diff / true_positives_diff

            # We now create new points TP_A + x for all integer values of x
            # such that 1 <= x <= TP_B + TP_A, i.e. TP_A + 1, .., TP_B - 1
            # and calculate the corresponding FP by linearly increasing the
            # false positives for each new point by the local skew
            C = A
            while abs(C.true_positives - B.true_positives) > 1.:
                interp_fp = A.false_positives + ((C.true_positives - A.true_positives) + 1) * local_skew
                C = PNPoint(C.true_positives + 1, interp_fp) if interp_fp > .0 else PNPoint(.0, .0)
                interpolated_points += [C]

        interpolated_points += [sorted_pn_points[-1]]

        self.pn_points = interpolated_points

    def calculate_auc_pr(self, min_recall=.0):
        thr = min_recall * self.nb_positives

        area, start = None, False

        for idx, A in enumerate(self.pn_points[:-1]):

            if A.true_positives >= thr:
                start = True

            if start is True:
                recall_A = A.true_positives / self.nb_positives
                precision_A = A.true_positives / (A.true_positives + A.false_positives)

                if area is None:
                    _recall_A = ((A.true_positives - thr) / self.nb_positives)
                    area = _recall_A * precision_A

                    if idx > 0:
                        C = self.pn_points[idx - 1]
                        recall_C = C.true_positives / self.nb_positives
                        precision_C = C.true_positives / (C.true_positives + C.false_positives)
                        ratio = (precision_A - precision_C) / (recall_A - recall_C)
                        update = precision_C + ratio * (thr - C.true_positives) / self.nb_positives
                        area += .5 * _recall_A * (update - precision_A)

                B = self.pn_points[idx + 1]
                recall_B = B.true_positives / self.nb_positives
                precision_B = B.true_positives / (B.true_positives + B.false_positives)

                area += .5 * (recall_B - recall_A) * (precision_A + precision_B)
        return area

    def calculate_auc_roc(self):
        area = None

        for idx, A in enumerate(self.pn_points[:-1]):

            true_positive_rate_A = A.true_positives / self.nb_positives
            false_positive_rate_A = A.false_positives / self.nb_negatives

            if area is None:
                area = .5 * true_positive_rate_A * false_positive_rate_A

            B = self.pn_points[idx + 1]

            true_positive_rate_B = B.true_positives / self.nb_positives
            false_positive_rate_B = B.false_positives / self.nb_negatives

            # (b - a) * (f(a) + f(b)) * .5
            area += .5 * (true_positive_rate_B - true_positive_rate_A) * (false_positive_rate_B + false_positive_rate_A)

        return 1. - area
