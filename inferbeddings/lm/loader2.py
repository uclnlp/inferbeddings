# -*- coding: utf-8 -*-

import gzip
import json

import numpy as np
import nltk

import logging

logger = logging.getLogger(__name__)


class SNLILoader:
    def __init__(self,
                 path='data/snli/snli_1.0_train.jsonl.gz',
                 batch_size=32, seq_length=8,
                 token_to_index=None,
                 bos_idx=1, eos_idx=2, unk_idx=3, seed=0, shuffle=True):

        assert token_to_index is not None
        assert path is not None

        self.path = path

        self.batch_size, self.seq_length = batch_size, seq_length
        self.token_to_index = token_to_index

        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.unk_idx = unk_idx

        self.seed = seed
        self.shuffle = shuffle

        self.random_state = np.random.RandomState(self.seed)

        self.sentences = SNLILoader.read_from_path(self.path)

        self.create_batches()
        self.reset_batch_pointer()

    @staticmethod
    def read_from_path(path):
        res = []
        with gzip.open(path, 'rb') as f:
            for line in f:
                decoded_line = line.decode('utf-8')
                obj = json.loads(decoded_line)
                s1, s2, gl = SNLILoader.extract_sentences(obj)
                if gl in {'entailment', 'neutral', 'contradiction'}:
                    res += [s1, s2]
        return res

    def create_batches(self):
        self.text_idxs = []
        order = self.random_state.permutation(len(self.sentences))

        for i in order:
            sentence = self.sentences[i]
            for word in sentence:
                self.text_idxs += [self.token_to_index.get(word, self.unk_idx)]

        self.tensor = np.array(self.text_idxs)
        self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))

        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]

        x_data = self.tensor
        y_data = np.copy(self.tensor)

        y_data[:-1] = x_data[1:]
        y_data[-1] = x_data[0]

        x_batches = np.split(x_data.reshape(self.batch_size, -1), self.num_batches, 1)
        y_batches = np.split(y_data.reshape(self.batch_size, -1), self.num_batches, 1)

        self.batches = [{'x': x, 'y': y} for x, y in zip(x_batches, y_batches)]
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
