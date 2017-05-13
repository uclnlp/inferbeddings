# -*- coding: utf-8 -*-

import abc
import numpy as np

from sklearn import metrics

from inferbeddings.evaluation.util import apk

import logging

logger = logging.getLogger(__name__)


class BaseRanker(metaclass=abc.ABCMeta):
    def __call__(self, pos_triples, neg_triples=None):
        raise NotImplementedError


class MeanAveragePrecision(BaseRanker):
    def __init__(self, scoring_function):
        self.scoring_function = scoring_function

    def __call__(self, pos_triples, neg_triples=None):
        # First, create a list wih all relation indices
        p_idxs = sorted({p for (_, p, _) in pos_triples + (neg_triples if neg_triples else [])})

        average_precisions = []

        # Iterate over each predicate p and create a list of positive and a list of negative triples for p
        for p_idx in p_idxs:
            p_pos_triples = [(s, p, o) for (s, p, o) in pos_triples if p == p_idx]
            p_neg_triples = [(s, p, o) for (s, p, o) in neg_triples if p == p_idx]

            # Score such triples:
            n = len(p_pos_triples + p_neg_triples)
            Xr = np.full(shape=(n, 1), fill_value=p_idx, dtype=np.int32)
            Xe = np.full(shape=(n, 2), fill_value=0, dtype=np.int32)

            for i, (s_idx, _p_idx, o_idx) in enumerate(p_pos_triples + p_neg_triples):
                assert _p_idx == p_idx
                Xe[i, 0], Xe[i, 1] = s_idx, o_idx

            scores = self.scoring_function([Xr, Xe])

            actual = range(1, len(p_pos_triples) + 1)
            predicted = 1 + np.argsort(scores)[::-1]

            average_precision = apk(actual=actual, predicted=predicted, k=n)
            average_precisions += [average_precision]
        return np.mean(average_precisions)


class Ranker(BaseRanker):
    def __init__(self, scoring_function, nb_entities, true_triples=None):
        self.scoring_function = scoring_function
        self.nb_entities = nb_entities
        self.true_triples = true_triples

    def __call__(self, pos_triples, neg_triples=None):
        err_subj, err_obj = [], []
        filtered_err_subj, filtered_err_obj = [], []

        for subj_idx, pred_idx, obj_idx in pos_triples:
            Xr = np.full(shape=(self.nb_entities, 1), fill_value=pred_idx, dtype=np.int32)

            Xe_o = np.full(shape=(self.nb_entities, 2), fill_value=obj_idx, dtype=np.int32)
            Xe_o[:, 0] = np.arange(1, self.nb_entities + 1)

            Xe_s = np.full(shape=(self.nb_entities, 2), fill_value=subj_idx, dtype=np.int32)
            Xe_s[:, 1] = np.arange(1, self.nb_entities + 1)

            # scores of (1, p, o), (2, p, o), .., (N, p, o)
            scores_o = self.scoring_function([Xr, Xe_o])

            # scores of (s, p, 1), (s, p, 2), .., (s, p, N)
            scores_s = self.scoring_function([Xr, Xe_s])

            #err_subj += [1 + np.sum(scores_o > scores_o[subj_idx - 1])]
            #err_obj += [1 + np.sum(scores_s > scores_s[obj_idx - 1])]

            err_subj += [1 + p.argsort(np.argsort(- scores_o))[subj_idx - 1]]
            err_obj += [1 + np.argsort(np.argsort(- scores_s))[obj_idx - 1]]

            if self.true_triples:
                rm_idx_o = [o - 1 for (s, p, o) in self.true_triples if s == subj_idx and p == pred_idx and o != obj_idx]
                rm_idx_s = [s - 1 for (s, p, o) in self.true_triples if o == obj_idx and p == pred_idx and s != subj_idx]

                if rm_idx_o:
                    scores_s[rm_idx_o] = - np.inf

                if rm_idx_s:
                    scores_o[rm_idx_s] = - np.inf

            #filtered_err_subj += [1 + np.sum(scores_o > scores_o[subj_idx - 1])]
            #filtered_err_obj += [1 + np.sum(scores_s > scores_s[obj_idx - 1])]

            filtered_err_subj += [1 + np.argsort(np.argsort(- scores_o))[subj_idx - 1]]
            filtered_err_obj += [1 + np.argsort(np.argsort(- scores_s))[obj_idx - 1]]

        return (err_subj, err_obj), (filtered_err_subj, filtered_err_obj)


class AUC(BaseRanker):
    def __init__(self, scoring_function, nb_entities, nb_predicates, rescale_predictions=False):
        self.scoring_function = scoring_function
        self.nb_entities = nb_entities
        self.nb_predicates = nb_predicates
        self.rescale_predictions = rescale_predictions

    def __call__(self, pos_triples, neg_triples=None):
        triples = pos_triples + neg_triples
        labels = [1 for _ in range(len(pos_triples))] + [0 for _ in range(len(neg_triples))]

        Xr, Xe = [], []
        for (s_idx, p_idx, o_idx), label in zip(triples, labels):
            Xr += [[p_idx]]
            Xe += [[s_idx, o_idx]]

        ascores = self.scoring_function([Xr, Xe])
        ays = np.array(labels)

        if self.rescale_predictions:
            diffs = np.diff(np.sort(ascores))
            min_diff = min(abs(diffs[np.nonzero(diffs)]))

            if min_diff < 1e-8:
                ascores = (ascores * (1e-7 / min_diff)).astype(np.float64)

        aucroc_value = metrics.roc_auc_score(ays, ascores)
        precision, recall, thresholds = metrics.precision_recall_curve(ays, ascores, pos_label=1)
        aucpr_value = metrics.auc(recall, precision)

        return aucroc_value, aucpr_value
