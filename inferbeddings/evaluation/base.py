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


def evaluate_auc(scoring_function, pos_triples, neg_triples, nb_entities, nb_predicates, tag=None):
    auc = metrics.AUC(scoring_function=scoring_function, nb_entities=nb_entities, nb_predicates=nb_predicates)

    aucroc_value, aucpr_value = auc(pos_triples, neg_triples)

    logger.info('[{}]\tAUC-ROC: {}'.format(tag, aucroc_value))
    logger.info('[{}]\tAUC-PR: {}'.format(tag, aucpr_value))

    return aucroc_value, aucpr_value


def evaluate_ranks(scoring_function, triples, nb_entities, true_triples=None, tag=None):
    if true_triples is None:
        true_triples = []

    ranker = metrics.Ranker(scoring_function=scoring_function, nb_entities=nb_entities, true_triples=true_triples)
    ranks, ranks_f = ranker(triples)

    if tag is not None:
        for n in range(1, 10 + 1):
            ranking_summary(ranks, n=n, tag='{} raw'.format(tag))

    if tag is not None:
        for n in range(1, 10 + 1):
            ranking_summary(ranks_f, n=n, tag='{} filtered'.format(tag))
    return ranks
