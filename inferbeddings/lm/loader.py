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
                 token_to_index=None):

        assert token_to_index is not None
        assert path is not None

        self.path = path
        self.token_to_index = token_to_index

        self.sentences = []

        with gzip.open(self.path, 'rb') as f:
            for line in f:
                decoded_line = line.decode('utf-8')
                obj = json.loads(decoded_line)

                s1, s2, gl = SNLILoader.extract_sentences(obj)

                if gl in {'entailment', 'neutral', 'contradiction'}:
                    self.sentences += [s1, s2]

        self.tensor = np.array(list(map(self.token_to_index.get, self.sentences)))




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

    loader = SNLILoader(token_to_index=token_to_index)
