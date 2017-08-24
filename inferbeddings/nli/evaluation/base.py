# -*- coding: utf-8 -*-

import numpy as np

from inferbeddings.models.training.util import make_batches

import logging

logger = logging.getLogger(__name__)


def accuracy(session, dataset, name,
             sentence1_ph, sentence1_length_ph, sentence2_ph, sentence2_length_ph, label_ph, dropout_keep_prob_ph,
             predictions_int, labels_int, contradiction_idx, entailment_idx, neutral_idx, batch_size):

    nb_eval_instances = len(dataset['sentence1'])
    eval_batches = make_batches(size=nb_eval_instances, batch_size=batch_size)
    p_vals, l_vals = [], []

    for e_batch_start, e_batch_end in eval_batches:

        feed_dict = {
            sentence1_ph: dataset['sentence1'][e_batch_start:e_batch_end],
            sentence1_length_ph: dataset['sentence1_length'][e_batch_start:e_batch_end],

            sentence2_ph: dataset['sentence2'][e_batch_start:e_batch_end],
            sentence2_length_ph: dataset['sentence2_length'][e_batch_start:e_batch_end],

            label_ph: dataset['label'][e_batch_start:e_batch_end],

            dropout_keep_prob_ph: 1.0
        }

        p_val, l_val = session.run([predictions_int, labels_int], feed_dict=feed_dict)

        p_vals += p_val.tolist()
        l_vals += l_val.tolist()

    matches = np.equal(p_vals, l_vals)
    acc = np.mean(matches)

    acc_c = np.mean(matches[np.where(np.array(l_vals) == contradiction_idx)])
    acc_e = np.mean(matches[np.where(np.array(l_vals) == entailment_idx)])
    acc_n = np.mean(matches[np.where(np.array(l_vals) == neutral_idx)])

    if name:
        logger.debug('{0} Accuracy: {1:.4f} - C: {2:.4f}, E: {3:.4f}, N: {4:.4f}'.format(
            name, acc * 100, acc_c * 100, acc_e * 100, acc_n * 100))

    return acc, acc_c, acc_e, acc_n
