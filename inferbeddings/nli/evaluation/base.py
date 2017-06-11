# -*- coding: utf-8 -*-

import numpy as np

from inferbeddings.models.training.util import make_batches

import logging

logger = logging.getLogger(__name__)


def accuracy(session, dataset,
             sentence1_ph, sentence1_length_ph, sentence2_ph, sentence2_length_ph, label_ph, dropout_keep_prob_ph,
             predictions_int, labels_int,
             contradiction_idx, entailment_idx, neutral_idx,
             batch_size):

    nb_eval_instances = len(dataset['questions'])
    eval_batches = make_batches(size=nb_eval_instances, batch_size=batch_size)
    p_vals, l_vals = [], []

    for e_batch_start, e_batch_end in eval_batches:

        feed_dict = {
            sentence1_ph: dataset['questions'][e_batch_start:e_batch_end],
            sentence2_ph: dataset['supports'][e_batch_start:e_batch_end],
            sentence1_length_ph: dataset['question_lengths'][e_batch_start:e_batch_end],
            sentence2_length_ph: dataset['support_lengths'][e_batch_start:e_batch_end],
            label_ph: dataset['answers'][e_batch_start:e_batch_end],
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

    return acc, acc_c, acc_e, acc_n
