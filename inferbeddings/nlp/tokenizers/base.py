# -*- coding: utf-8 -*-


class Tokenizer(object):
    def __init__(self, num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                 lower=True, split=' ', char_level=False,
                 has_bos=False, has_eos=False, has_unk=False):
        self.word_counts = dict()
        self.word_index = dict()
        self.filters = filters
        self.split = split
        self.lower = lower
        self.num_words = num_words
        self.char_level = char_level
        self.has_bos, self.has_eos, self.has_unk = has_bos, has_eos, has_unk
        self.bos_idx = self.eos_idx = self.unk_idx = None
        self.start_idx = None

    @staticmethod
    def text_to_word_seq(text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' '):
        if lower:
            text = text.lower()
        text = text.translate(str.maketrans(filters, split * len(filters)))
        return [token for token in text.split(split) if token]

    def fit_on_texts(self, texts):
        for text in texts:
            seq = text if self.char_level else Tokenizer.text_to_word_seq(text, self.filters, self.lower, self.split)
            for word in seq:
                if word not in self.word_counts:
                    self.word_counts[word] = 0
                self.word_counts[word] += 1

        sorted_voc = [word[0] for word in sorted(self.word_counts.items(), key=lambda kv: (- kv[1], kv[0]))]

        # note that index 0 is reserved, never assigned to an existing word
        self.start_idx = 1

        if self.has_bos:
            self.bos_idx = self.start_idx
            self.start_idx += 1

        if self.has_eos:
            self.eos_idx = self.start_idx
            self.start_idx += 1

        if self.has_unk:
            self.unk_idx = self.start_idx
            self.start_idx += 1

        indices = list(range(self.start_idx, len(sorted_voc) + self.start_idx))
        self.word_index = dict(list(zip(sorted_voc, indices)))

    def texts_to_sequences(self, texts):
        return [vector for vector in self.texts_to_sequences_generator(texts)]

    def texts_to_sequences_generator(self, texts):
        num_words = self.num_words
        for text in texts:
            seq = text if self.char_level else Tokenizer.text_to_word_seq(text, self.filters, self.lower, self.split)
            vector = [self.bos_idx] if self.has_bos else []
            for word in seq:
                idx = self.word_index.get(word)
                if idx and not (num_words and idx >= num_words):
                    vector += [idx]
                else:
                    vector += [self.unk_idx] if self.has_unk else []
            vector += [self.eos_idx] if self.has_eos else []
            yield vector
