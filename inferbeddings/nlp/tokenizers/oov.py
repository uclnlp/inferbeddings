# -*- coding: utf-8 -*-

import logging

logger = logging.getLogger(__name__)


class OOVTokenizer(object):
    def __init__(self,
                 num_words=None,
                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                 lower=True,
                 split=' ',
                 has_bos=False, has_eos=False, has_unk=False,
                 vocabulary=None):
        self.word_counts = dict()
        self.word_index = dict()
        self.filters = filters
        self.split = split
        self.lower = lower
        self.num_words = num_words

        self.has_bos, self.has_eos, self.has_unk = has_bos, has_eos, has_unk
        self.vocabulary = vocabulary or set()

        self.bos_idx = self.eos_idx = self.unk_idx = None

    @staticmethod
    def text_to_word_seq(text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' '):
        if lower:
            text = text.lower()
        text = text.translate(str.maketrans(filters, split * len(filters)))
        return [i for i in text.split(split) if i]

    def fit_on_texts(self, texts):
        for text in texts:
            seq = OOVTokenizer.text_to_word_seq(text, self.filters, self.lower, self.split)
            for word in seq:
                if word not in self.word_counts:
                    self.word_counts[word] = 0
                self.word_counts[word] += 1

        word_iv_counts_lst = [(word, word in self.vocabulary, count) for word, count in self.word_counts.items()]

        print(word_iv_counts_lst)

        sorted_voc = [word[0] for word in sorted(word_iv_counts_lst, key=lambda wic: (wic[1], - wic[2], wic[0]))]

        print(sorted_voc)

        # note that index 0 is reserved, never assigned to an existing word
        start_idx = 1

        if self.has_bos:
            self.bos_idx = start_idx
            start_idx += 1

        if self.has_eos:
            self.eos_idx = start_idx
            start_idx += 1

        if self.has_unk:
            self.unk_idx = start_idx
            start_idx += 1

        # note that index 0 is reserved, never assigned to an existing word
        indices = list(range(start_idx, len(sorted_voc) + start_idx))
        self.word_index = dict(list(zip(sorted_voc, indices)))

    def texts_to_sequences(self, texts):
        return [vector for vector in self.texts_to_sequences_generator(texts)]

    def texts_to_sequences_generator(self, texts):
        num_words = self.num_words
        for text in texts:
            seq = OOVTokenizer.text_to_word_seq(text, self.filters, self.lower, self.split)
            vector = [self.bos_idx] if self.has_bos else []
            for word in seq:
                idx = self.word_index.get(word)
                if idx and not (num_words and idx >= num_words):
                    vector += [idx]
                else:
                    vector += [self.unk_idx] if self.has_unk else []
            vector += [self.eos_idx] if self.has_eos else []
            yield vector
