# -*- coding: utf-8 -*-

import gzip
import json

import numpy as np
import nltk

from inferbeddings.nli.util import pad_sequences
from inferbeddings.models.training.util import make_batches

import logging

logger = logging.getLogger(__name__)


class SNLILoader:
    def __init__(self,
                 path='data/snli/snli_1.0_train.jsonl.gz',
                 batch_size=32, seq_length=8,
                 token_to_index=None,
                 bos_idx=1, eos_idx=2, unk_idx=3, seed=0):

        assert token_to_index is not None
        assert path is not None

        self.path = path
        self.batch_size, self.seq_length = batch_size, seq_length
        self.token_to_index = token_to_index

        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.unk_idx = unk_idx

        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)

        self.sentences = []

        with gzip.open(self.path, 'rb') as f:
            for line in f:
                decoded_line = line.decode('utf-8')
                obj = json.loads(decoded_line)

                s1, s2, gl = SNLILoader.extract_sentences(obj)

                if gl in {'entailment', 'neutral', 'contradiction'}:
                    self.sentences += [s1, s2]

        self.sentence_idxs = []
        for sentence in self.sentences:
            s_idxs = [self.token_to_index.get(word, self.unk_idx) for word in sentence]
            self.sentence_idxs += [s_idxs]

        self.tensor = pad_sequences(self.sentence_idxs)
        self.nb_samples, self.max_len = self.tensor.shape
        self.pointer = 0

    def create_batches(self):
        order = self.random_state.permutation(self.nb_samples)
        tensor_shuf = self.tensor[order, :]

        _batch_lst = make_batches(self.nb_samples, self.batch_size)
        self.batches = []

        for batch_start, batch_end in _batch_lst:
            batch_size = batch_end - batch_start
            batch = tensor_shuf[batch_start:batch_end, :]

            assert batch.shape[0] == batch_size

            x = np.zeros(shape=(batch_size, self.seq_length))
            y = np.zeros(shape=(batch_size, self.seq_length))

            for i in range(batch_size):
                start_idx = self.random_state.randint(low=0, high=self.max_len - 1)
                end_idx = min(start_idx + self.seq_length, self.max_len)

                x[i, 0:(end_idx - start_idx)] = batch[i, start_idx:end_idx]

                start_idx += 1
                end_idx = min(start_idx + self.seq_length, self.max_len)

                y[i, 0:(end_idx - start_idx)] = batch[i, start_idx:end_idx]

                d = {
                    'x': x,
                    'y': y
                }
                self.batches += [d]

        self.num_batches = len(self.batches)
        return

    def next_batch(self):
        batch = self.batches[self.pointer]
        self.pointer += 1
        return batch['x'], batch['y']

    def reset_batch_pointer(self):
        self.pointer = 0
        return

    @staticmethod
    def extract_sentences(obj):
        sentence1_parse = obj['sentence1_parse']
        sentence1_tree = nltk.Tree.fromstring(sentence1_parse)
        sentence1_parse_tokens = sentence1_tree.leaves()

        sentence2_parse = obj['sentence2_parse']
        sentence2_tree = nltk.Tree.fromstring(sentence2_parse)
        sentence2_parse_tokens = sentence2_tree.leaves()

        gold_label = obj['gold_label']
        return sentence1_parse_tokens, sentence2_parse_tokens, gold_label


if __name__ == '__main__':
    import pickle

    path = 'models/snli/dam_1/dam_1_index_to_token.p'
    with open(path, 'rb') as f:
        index_to_token = pickle.load(f)

    token_to_index = {token: index for index, token in index_to_token.items()}
    loader = SNLILoader(path='data/snli/snli_1.0_train.jsonl.gz', token_to_index=token_to_index)

    print(loader.tensor.shape)
    print(loader.tensor)

    print(np.nonzero(loader.tensor[:, -1]))
