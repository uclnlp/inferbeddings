# -*- coding: utf-8 -*-

import os
import codecs
import collections
import numpy as np
import pickle

import logging

logger = logging.getLogger(__name__)


class TextLoader:
    def __init__(self, data_dir, batch_size, seq_length, encoding=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        # Let's not read vocab and data from file. We many change them.
        if True or not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            logger.info("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file, encoding)
        else:
            logger.info("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def build_vocabulary(self, sentences):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = collections.Counter(sentences)

        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        vocabulary_inv = list(sorted(vocabulary_inv))

        # Mapping from word to index
        vocabulary = {word: idx for idx, word in enumerate(vocabulary_inv)}

        return vocabulary, vocabulary_inv

    def preprocess(self, input_file, vocab_file, tensor_file, encoding):
        with codecs.open(input_file, "r", encoding=encoding) as f:
            data = f.read()

        # Optional text cleaning or make them lower case, etc.
        x_text = data.split()

        self.vocab, self.words = self.build_vocabulary(x_text)
        self.vocab_size = len(self.words)

        with open(vocab_file, 'wb') as f:
            pickle.dump(self.words, f)

        # The same operation like this [self.vocab[word] for word in x_text]
        # index of words as our basic data
        self.tensor = np.array(list(map(self.vocab.get, x_text)))
        # Save the data to data.npy
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.words = pickle.load(f)

        self.vocab_size = len(self.words)
        self.vocab = dict(zip(self.words, range(len(self.words))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))
        return

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)

        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]

        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)
        return

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
        return
