# -*- coding: utf-8 -*-


class Tokenizer(object):
    def __init__(self, num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                 lower=True, split=' ', char_level=False):
        self.word_counts = dict()
        self.word_index = dict()
        self.filters = filters
        self.split = split
        self.lower = lower
        self.num_words = num_words
        self.document_count = 0
        self.char_level = char_level

    @staticmethod
    def text_to_word_seq(text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" "):
        if lower:
            text = text.lower()
        text = text.translate(str.maketrans(filters, split * len(filters)))
        seq = text.split(split)
        return [i for i in seq if i]

    def fit_on_texts(self, texts):
        self.document_count = 0
        for text in texts:
            self.document_count += 1
            seq = text if self.char_level else Tokenizer.text_to_word_seq(text, self.filters, self.lower, self.split)
            for w in seq:
                if w not in self.word_counts:
                    self.word_counts[w] = 0
                self.word_counts[w] += 1

        sorted_voc = [word[0] for word in sorted(self.word_counts.items(), key=lambda kv: (- kv[1], kv[0]))]

        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))

    def texts_to_sequences(self, texts):
        return [v for v in self.texts_to_sequences_generator(texts)]

    def texts_to_sequences_generator(self, texts):
        num_words = self.num_words
        for text in texts:
            seq = text if self.char_level else Tokenizer.text_to_word_seq(text, self.filters, self.lower, self.split)
            v = []
            for w in seq:
                idx = self.word_index.get(w)
                if idx and not (num_words and idx >= num_words):
                    v += [idx]
            yield v

