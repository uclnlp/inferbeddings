# -*- coding: utf-8 -*-

import numpy as np
import inferbeddings.nli.util as util
import logging

import pytest

logger = logging.getLogger(__name__)


def get_train(has_bos, has_eos, has_unk):
    path = 'data/snli/tiny/tiny.jsonl.gz'
    train_instances, dev_instances, test_instances =\
        util.SNLI.generate(train_path=path, valid_path=path, test_path=path)
    all_instances = train_instances + dev_instances + test_instances

    # Create a sequence of tokens containing all sentences in the dataset
    token_seq = []
    for instance in all_instances:
        token_seq += instance['sentence1_parse_tokens'] + instance['sentence2_parse_tokens']

    # Count the number of occurrences of each token
    token_counts = dict()
    for token in token_seq:
        if token not in token_counts:
            token_counts[token] = 0
        token_counts[token] += 1

    # Sort the tokens according to their frequency and lexicographic ordering
    sorted_vocabulary = sorted(token_counts.keys(), key=lambda t: (- token_counts[t], t))

    # Enumeration of tokens start at index=3:
    # index=0 PADDING, index=1 START_OF_SENTENCE, index=2 END_OF_SENTENCE, index=3 UNKNOWN_WORD
    bos_idx, eos_idx, unk_idx = 1, 2, 3
    start_idx = 1 + (1 if has_bos else 0) + (1 if has_eos else 0) + (1 if has_unk else 0)

    index_to_token = {index: token for index, token in enumerate(sorted_vocabulary, start=start_idx)}
    token_to_index = {token: index for index, token in index_to_token.items()}

    entailment_idx, neutral_idx, contradiction_idx = 0, 1, 2
    label_to_index = {
        'entailment': entailment_idx,
        'neutral': neutral_idx,
        'contradiction': contradiction_idx
    }

    max_len = None
    train_dataset = util.instances_to_dataset(train_instances, token_to_index, label_to_index,
                                              has_bos=has_bos, has_eos=has_eos, has_unk=has_unk,
                                              bos_idx=bos_idx, eos_idx=eos_idx, unk_idx=unk_idx,
                                              max_len=max_len)
    return train_dataset


@pytest.mark.light
def test_nli_util():
    has_bos, has_eos, has_unk = True, True, True
    train_dataset_v1 = get_train(has_bos=has_bos, has_eos=has_eos, has_unk=has_unk)

    has_bos, has_eos, has_unk = False, False, False
    train_dataset_v2 = get_train(has_bos=has_bos, has_eos=has_eos, has_unk=has_unk)

    np.testing.assert_allclose(np.array(train_dataset_v2['sentence1_length']) + 2, train_dataset_v1['sentence1_length'])
    np.testing.assert_allclose(np.array(train_dataset_v2['sentence2_length']) + 2, train_dataset_v1['sentence2_length'])

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    pytest.main([__file__])
