# -*- coding: utf-8 -*-

import abc
import numpy as np

import itertools
from sklearn import metrics

import logging

logger = logging.getLogger(__name__)


class BaseRanker(metaclass=abc.ABCMeta):
    def __call__(self, pos_triples, neg_triples=None):
        raise NotImplementedError


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

            scores_o, scores_s = self.scoring_function([Xr, Xe_o]), self.scoring_function([Xr, Xe_s])

            err_subj += [1 + np.sum(scores_o > scores_o[subj_idx - 1])]
            err_obj += [1 + np.sum(scores_s > scores_s[obj_idx - 1])]

            if self.true_triples:
                rm_idx_o = [o - 1 for (s, p, o) in self.true_triples if s == subj_idx and p == pred_idx and o != obj_idx]
                rm_idx_s = [s - 1 for (s, p, o) in self.true_triples if o == obj_idx and p == pred_idx and s != subj_idx]

                if rm_idx_o:
                    scores_s[rm_idx_o] = - np.inf

                if rm_idx_s:
                    scores_o[rm_idx_s] = - np.inf

            filtered_err_subj += [1 + np.sum(scores_o > scores_o[subj_idx - 1])]
            filtered_err_obj += [1 + np.sum(scores_s > scores_s[obj_idx - 1])]

        return (err_subj, err_obj), (filtered_err_subj, filtered_err_obj)


class AUC(BaseRanker):
    def __init__(self, scoring_function, nb_entities, nb_predicates, rescale_predictions=True):
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
