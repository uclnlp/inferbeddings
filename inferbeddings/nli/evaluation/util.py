# -*- coding: utf-8 -*-

import json
import gzip

import numpy as np

import nltk

from inferbeddings.nli import util
from inferbeddings.models.training.util import make_batches


def evaluate(session, eval_path, label_to_index, token_to_index, predictions_op, batch_size,
             sentence1_ph, sentence2_ph, sentence1_len_ph, sentence2_len_ph, dropout_keep_prob_ph,
             has_bos=False, has_eos=False, has_unk=False, is_lower=False,
             bos_idx=1, eos_idx=2, unk_idx=3):
    sentence1_all = []
    sentence2_all = []
    gold_label_all = []

    with gzip.open(eval_path, 'rb') as f:
        for line in f:
            decoded_line = line.decode('utf-8')

            if is_lower:
                decoded_line = decoded_line.lower()

            obj = json.loads(decoded_line)

            gold_label = obj['gold_label']

            if gold_label in ['contradiction', 'entailment', 'neutral']:
                gold_label_all += [label_to_index[gold_label]]

                sentence1_parse = obj['sentence1_parse']
                sentence2_parse = obj['sentence2_parse']

                sentence1_tree = nltk.Tree.fromstring(sentence1_parse)
                sentence2_tree = nltk.Tree.fromstring(sentence2_parse)

                sentence1_tokens = sentence1_tree.leaves()
                sentence2_tokens = sentence2_tree.leaves()

                sentence1_ids = []
                sentence2_ids = []

                if has_bos:
                    sentence1_ids += [bos_idx]
                    sentence2_ids += [bos_idx]

                for token in sentence1_tokens:
                    if token in token_to_index:
                        sentence1_ids += [token_to_index[token]]
                    elif has_unk:
                        sentence1_ids += [unk_idx]

                for token in sentence2_tokens:
                    if token in token_to_index:
                        sentence2_ids += [token_to_index[token]]
                    elif has_unk:
                        sentence2_ids += [unk_idx]

                if has_eos:
                    sentence1_ids += [eos_idx]
                    sentence2_ids += [eos_idx]

                sentence1_all += [sentence1_ids]
                sentence2_all += [sentence2_ids]

    sentence1_all_len = [len(s) for s in sentence1_all]
    sentence2_all_len = [len(s) for s in sentence2_all]

    np_sentence1 = util.pad_sequences(sequences=sentence1_all)
    np_sentence2 = util.pad_sequences(sequences=sentence2_all)

    np_sentence1_len = np.array(sentence1_all_len)
    np_sentence2_len = np.array(sentence2_all_len)

    gold_label = np.array(gold_label_all)

    nb_instances = gold_label.shape[0]
    batches = make_batches(size=nb_instances, batch_size=batch_size)

    predictions = []

    for batch_idx, (batch_start, batch_end) in enumerate(batches):
        feed_dict = {
            sentence1_ph: np_sentence1[batch_start:batch_end],
            sentence2_ph: np_sentence2[batch_start:batch_end],

            sentence1_len_ph: np_sentence1_len[batch_start:batch_end],
            sentence2_len_ph: np_sentence2_len[batch_start:batch_end],

            dropout_keep_prob_ph: 1.0
        }

        _predictions = session.run(predictions_op, feed_dict=feed_dict)
        predictions += _predictions.tolist()

    matches = np.array(predictions) == gold_label
    return np.mean(matches)
