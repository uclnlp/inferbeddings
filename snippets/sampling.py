# -*- coding: utf-8 -*-

import sys
import numpy as np
import logging


class ContrastiveTrainingProvider(object):
    def __init__(self, rng, train, entities):
        self.rng = rng
        self.train = train
        self.entities = entities

        self.num_examples = len(train)
        self.index_in_epoch = 0

        # store set of training tuples for quickly checking negatives
        self.triples_set = set(tuple(t) for t in train)

        # replacement entities
        head_replacements = list(set(train[:, 0]))
        tail_replacements = list(set(train[:, 2]))

        self.field_replacements = [head_replacements, list(set(train[:, 1])), tail_replacements]
        self.rng.shuffle(self.train)

    @staticmethod
    def corrupt(rng, triple, field_replacements, fields):
        field = rng.choice(fields)
        replacements = field_replacements[field]
        corrupted = list(triple)
        corrupted[field] = replacements[rng.randint(len(replacements))]
        return corrupted

    # TODO: Check if we can replace with custom iterator
    def next_batch(self):
        start = self.index_in_epoch
        self.index_in_epoch += self.batch_pos_cnt

        if self.index_in_epoch > self.num_examples:
            self.index_in_epoch = self.batch_pos_cnt
            start = 0
            self.rng.shuffle(self.train)
        end = self.index_in_epoch

        batch_triples = []
        batch_labels = []

        for positive in self.train[start:end]:
            batch_triples += [positive]
            batch_labels += [1.0]

            negative = ContrastiveTrainingProvider.corrupt(self.rng, positive, self.field_replacements, fields=[1, 2])
            batch_triples += [negative]
            batch_labels += [0.0]

        print(batch_triples)

        return np.vstack(batch_triples), np.array(batch_labels)


def main(argv):
    train = np.array([[1, 2, 3], [4, 2, 6]])
    ct = ContrastiveTrainingProvider(np.random.RandomState(0), train=train)
    print(ct.next_batch())

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])

