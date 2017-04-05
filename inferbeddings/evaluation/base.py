# -*- coding: utf-8 -*-

import numpy as np

from inferbeddings.evaluation import metrics
import logging


logger = logging.getLogger(__name__)


def ranking_summary(res, n=10, tag=None):
    dres = dict()

    dres['microlmean'] = np.mean(res[0])
    dres['microlmedian'] = np.median(res[0])
    dres['microlhits@n'] = np.mean(np.asarray(res[0]) <= n) * 100
    dres['micrormean'] = np.mean(res[1])
    dres['micrormedian'] = np.median(res[1])
    dres['microrhits@n'] = np.mean(np.asarray(res[1]) <= n) * 100

    resg = res[0] + res[1]

    dres['microgmean'] = np.mean(resg)
    dres['microgmedian'] = np.median(resg)
    dres['microghits@n'] = np.mean(np.asarray(resg) <= n) * 100

    dres['microlmrr'] = np.mean(1. / np.asarray(res[0]))
    dres['micrormrr'] = np.mean(1. / np.asarray(res[1]))
    dres['microgmrr'] = np.mean(1. / np.asarray(resg))

    logger.info('### MICRO (%s):' % tag)
    logger.info('\t-- left   >> mean: %s, median: %s, mrr: %s, hits@%s: %s%%' %
                (round(dres['microlmean'], 5), round(dres['microlmedian'], 5),
                 round(dres['microlmrr'], 3), n, round(dres['microlhits@n'], 3)))
    logger.info('\t-- right  >> mean: %s, median: %s, mrr: %s, hits@%s: %s%%' %
                (round(dres['micrormean'], 5), round(dres['micrormedian'], 5),
                 round(dres['micrormrr'], 3), n, round(dres['microrhits@n'], 3)))
    logger.info('\t-- global >> mean: %s, median: %s, mrr: %s, hits@%s: %s%%' %
                (round(dres['microgmean'], 5), round(dres['microgmedian'], 5),
                 round(dres['microgmrr'], 3), n, round(dres['microghits@n'], 3)))


def evaluate_map(scoring_function, pos_triples, neg_triples, tag=None):
    _map = metrics.MeanAveragePrecision(scoring_function)
    map_value = _map(pos_triples, neg_triples)
    logger.info('[{}]\tMean Average Precision (MAP): {}'.format(tag, map_value))
    return map_value


def evaluate_auc(scoring_function, pos_triples, neg_triples, nb_entities, nb_predicates, tag=None):
    auc = metrics.AUC(scoring_function=scoring_function, nb_entities=nb_entities, nb_predicates=nb_predicates)
    auc_roc_value, auc_pr_value = auc(pos_triples, neg_triples)
    logger.info('[{}]\tAUC-ROC: {}'.format(tag, auc_roc_value))
    logger.info('[{}]\tAUC-PR: {}'.format(tag, auc_pr_value))
    return auc_roc_value, auc_pr_value


def evaluate_ranks(scoring_function, triples, nb_entities, true_triples=None, tag=None,
                   verbose=False, index_to_predicate=None):
    if true_triples is None:
        true_triples = []

    ranker = metrics.Ranker(scoring_function=scoring_function, nb_entities=nb_entities,
                            true_triples=true_triples)
    # ranks and ranks_filtered have the form (list, list):
    # the former (resp. latter) list is the ranks on triples obtained corrupting the subject (resp. object)
    ranks, ranks_filtered = ranker(triples)

    if tag is not None:
        for n in range(1, 10 + 1):
            ranking_summary(ranks, n=n, tag='{} raw'.format(tag))

    if tag is not None:
        for n in range(1, 10 + 1):
            ranking_summary(ranks_filtered, n=n, tag='{} filtered'.format(tag))

    if verbose:
        assert index_to_predicate is not None

        ranks_l, ranks_r = ranks
        ranks_l_filtered, ranks_r_filtered = ranks_filtered

        p_idxs = sorted({p for (_, p, _) in triples})

        p_to_ranks_l = {p_idx: [] for p_idx in p_idxs}
        p_to_ranks_l_filtered = {p_idx: [] for p_idx in p_idxs}

        p_to_ranks_r = {p_idx: [] for p_idx in p_idxs}
        p_to_ranks_r_filtered = {p_idx: [] for p_idx in p_idxs}

        for (_, p, _), r_l, r_r, r_l_f, r_r_f in zip(triples, ranks_l, ranks_r, ranks_l_filtered, ranks_r_filtered):
            p_to_ranks_l[p] += [r_l]
            p_to_ranks_r[p] += [r_r]

            p_to_ranks_l_filtered[p] += [r_l_f]
            p_to_ranks_r_filtered[p] += [r_r_f]

        for p_idx in p_idxs:
            predicate_name = index_to_predicate[p_idx]

            ranks = (p_to_ranks_l[p_idx], p_to_ranks_r[p_idx])
            ranks_f = (p_to_ranks_l_filtered[p_idx], p_to_ranks_r_filtered[p_idx])

            if tag is not None:
                for n in range(1, 10 + 1):
                    ranking_summary(ranks, n=n, tag='{}\t{} raw'.format(predicate_name, tag))

            if tag is not None:
                for n in range(1, 10 + 1):
                    ranking_summary(ranks_f, n=n, tag='{}\t{} filtered'.format(predicate_name, tag))

    return ranks
