# -*- coding: utf-8 -*-

import abc

import random


class AWalker(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    def __call__(self, length):
        while False:
            yield None


class BidirectionalWalker(AWalker):
    def __init__(self, triples, seed=None):
        super().__init__()
        self.triples = triples
        self.random_state = random.Random(seed if seed is not None else 0)

        self.entities = {s for (s, p, o) in triples} | {o for (s, p, o) in triples}
        self.predicates = {p for (s, p, o) in triples}

        self.entity_to_triples = {e: set() for e in self.entities}

        for (s, p, o) in self.triples:
            self.entity_to_triples[s].add((s, p, o))
            self.entity_to_triples[o].add((s, p, o))

    def __call__(self, length):
        # Sample a source entity
        source = self.random_state.sample(population=self.entities, k=1)[0]

        target = source
        e = source

        steps = []
        for i in range(length):
            e_triples = self.entity_to_triples[e]

            # Sample a predicate among its incoming and outgoing edges
            e_predicates = {p for (s, p, o) in e_triples}
            predicate = self.random_state.sample(population=e_predicates, k=1)[0]

            # Uniformly sample a triple involving entity e and predicate p
            e_p_triples = {(s, p, o) for (s, p, o) in e_triples if predicate == p}

            # Uniformly sample a new node e
            (s, p, o) = self.random_state.sample(population=e_p_triples, k=1)[0]
            (e, is_inverse) = (s, True) if e == o else (o, False)

            steps += [(predicate, is_inverse)]

            target = e

        return steps, [source, target]
