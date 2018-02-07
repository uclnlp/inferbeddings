# -*- coding: utf-8 -*-

import nltk
from nltk.tree import Tree
import numpy as np

from inferbeddings.nli.generate.parser import Parser
from inferbeddings.nli.generate import operators as O


class Generator:
    def __init__(self, token_to_index, corenlp_url='http://127.0.0.1:9000',
                 is_remove=True, is_merge=True,
                 nb_corruptions=32, nb_words=1024,
                 bos_idx=1, eos_idx=2, unk_idx=3,
                 seed=0):

        self.token_to_index = token_to_index
        self.index_to_token = {k: v for v, k in self.token_to_index.items()}
        self.unk_token = self.index_to_token[unk_idx]

        self.corenlp_url = corenlp_url
        self.is_remove = is_remove
        self.is_merge = is_merge
        self.nb_corruptions = nb_corruptions
        self.nb_words = nb_words

        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.unk_idx = unk_idx

        self.rs = np.random.RandomState(seed)

        self.parser = Parser(url=corenlp_url)
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()

        self.cache = {}

    def combine(self, sentence1, sentence2):
        sentence1_str, sentence2_str = sentence1, sentence2

        # If the input is arrays of indices, turn each array into a string first
        if isinstance(sentence1, list):
            sentence1_str = self._to_string(sentence1)
            sentence2_str = self._to_string(sentence2)

        # Parse the sentences
        tree1 = self._parse(sentence1_str)
        tree2 = self._parse(sentence2_str)

        tree1_variations = O.combine_trees(tree1, tree2)
        tree2_variations = O.combine_trees(tree1, tree2)

        # Remove duplicate trees, working around the issue that Tree is not hashable
        tree1_variations = [Tree.fromstring(s) for s in list(set([str(t) for t in tree1_variations]))]
        tree2_variations = [Tree.fromstring(s) for s in list(set([str(t) for t in tree2_variations]))]

        res1, res2 = [], []
        for idx in range(self.nb_corruptions):
            tree1_c, tree2_c = tree1, tree2
            if self.rs.randint(0, 2):
                tree1_c = tree1_variations[self.rs.randint(low=0, high=len(tree1_variations))]
            else:
                tree2_c = tree2_variations[self.rs.randint(low=0, high=len(tree2_variations))]

            res1 += [[self.token_to_index.get(tkn, self.unk_idx) for tkn in tree1_c.leaves()]]
            res2 += [[self.token_to_index.get(tkn, self.unk_idx) for tkn in tree2_c.leaves()]]

        # If the input was strings, the output should also be strings
        if isinstance(sentence1, str):
            new_res1 = [self._to_string(e) for e in res1]
            new_res2 = [self._to_string(e) for e in res2]
            res1, res2 = new_res1, new_res2

        return res1, res2

    def remove(self, sentence1, sentence2):
        sentence1_str, sentence2_str = sentence1, sentence2

        # If the input is arrays of indices, turn each array into a string first
        if isinstance(sentence1, list):
            sentence1_str = self._to_string(sentence1)
            sentence2_str = self._to_string(sentence2)

        # Parse the sentences
        tree1 = self._parse(sentence1_str)
        tree2 = self._parse(sentence2_str)

        tree1_variations = O.remove_subtree(tree1)
        tree2_variations = O.remove_subtree(tree2)

        # Remove duplicate trees, working around the issue that Tree is not hashable
        tree1_variations = [Tree.fromstring(s) for s in list(set([str(t) for t in tree1_variations]))]
        tree2_variations = [Tree.fromstring(s) for s in list(set([str(t) for t in tree2_variations]))]

        res1, res2 = [], []
        for idx in range(self.nb_corruptions):
            tree1_c, tree2_c = tree1, tree2
            if self.rs.randint(0, 2):
                tree1_c = tree1_variations[self.rs.randint(low=0, high=len(tree1_variations))]
            else:
                tree2_c = tree2_variations[self.rs.randint(low=0, high=len(tree2_variations))]

            res1 += [[self.token_to_index.get(tkn, self.unk_idx) for tkn in tree1_c.leaves()]]
            res2 += [[self.token_to_index.get(tkn, self.unk_idx) for tkn in tree2_c.leaves()]]

        # If the input was strings, the output should also be strings
        if isinstance(sentence1, str):
            new_res1 = [self._to_string(e) for e in res1]
            new_res2 = [self._to_string(e) for e in res2]
            res1, res2 = new_res1, new_res2

        return res1, res2

    def flip(self, sentence1, sentence2):
        # If the input is strings, turn each string in a list of token indices.
        if isinstance(sentence1, str):
            assert isinstance(sentence2, str)
            sentence1_tkns = self._tokenize(sentence1)
            sentence2_tkns = self._tokenize(sentence2)

            sentence1_idxs = [self.token_to_index.get(tkn, self.unk_idx) for tkn in sentence1_tkns]
            sentence2_idxs = [self.token_to_index.get(tkn, self.unk_idx) for tkn in sentence2_tkns]
        else:
            assert isinstance(sentence1, list) and isinstance(sentence2, list)
            sentence1_idxs = sentence1
            sentence2_idxs = sentence2

        corruptions1 = np.repeat(a=[sentence1_idxs], repeats=self.nb_corruptions, axis=0)
        corruptions2 = np.repeat(a=[sentence2_idxs], repeats=self.nb_corruptions, axis=0)

        sentence1_len, sentence2_len = len(sentence1_idxs), len(sentence2_idxs)

        for idx in range(self.nb_corruptions):
            new_word = self.rs.randint(low=3, high=self.nb_words)

            if self.rs.randint(0, 2):
                where_to_corrupt1 = self.rs.randint(low=1, high=sentence1_len - 1)
                corruptions1[idx, where_to_corrupt1] = new_word
            else:
                where_to_corrupt2 = self.rs.randint(low=1, high=sentence2_len - 1)
                corruptions2[idx, where_to_corrupt2] = new_word

        res1, res2 = corruptions1.tolist(), corruptions2.tolist()

        # If the input is strings, the outputs needs to be strings as well.
        if isinstance(sentence1, str):
            new_res1 = [self._to_string(e) for e in res1]
            new_res2 = [self._to_string(e) for e in res2]
            res1, res2 = new_res1, new_res2

        return res1, res2

    def _parse(self, sentence):
        s = self._parse_str(sentence)
        return Tree.fromstring(s)

    def _parse_str(self, sentence):
        if sentence not in self.cache:
            tree = self.parser.parse(sentence)
            s_tree = str(tree)
            self.cache[sentence] = s_tree
        return self.cache[sentence]

    def _to_string(self, sentence_idxs):
        return ' '.join([self.index_to_token.get(idx, self.unk_token) for idx in sentence_idxs])

    def _tokenize(self, sentence):
        return self.tokenizer.tokenize(sentence)
