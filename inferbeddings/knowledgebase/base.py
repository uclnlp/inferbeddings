# -*- coding: utf-8 -*-


class Fact:
    def __init__(self, predicate_name, argument_names):
        self.predicate_name = predicate_name
        self.argument_names = argument_names

    def __str__(self):
        return '{0!s}({1!s})'.format(repr(self.predicate_name), repr(self.argument_names))

    def __repr__(self):
        return '<Fact {0!r}({1!r})>'.format(repr(self.predicate_name), repr(self.argument_names))

    def __eq__(self, other):
        res = False
        if isinstance(other, Fact):
            res = (self.predicate_name == other.predicate_name) and (self.argument_names == other.argument_names)
        return res

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.predicate_name, tuple(self.argument_names)))


class KnowledgeBaseParser:
    def __init__(self, facts):
        self.entity_vocabulary, self.predicate_vocabulary = set(), set()

        for fact in facts:
            self.predicate_vocabulary.add(fact.predicate_name)
            for arg in fact.argument_names:
                self.entity_vocabulary.add(arg)

        self.entity_to_index, self.predicate_to_index = KnowledgeBaseParser._fit(self.entity_vocabulary,
                                                                                 self.predicate_vocabulary)

        self.index_to_entity = {idx: e for e, idx in self.entity_to_index.items()}
        self.index_to_predicate = {idx: p for p, idx in self.predicate_to_index.items()}

    @staticmethod
    def _fit(entity_vocabulary, predicate_vocabulary):
        """
        Required before using facts_to_sequences
        :param facts: List or generator of facts.
        :return:
        """
        sorted_ent_lst = sorted(entity_vocabulary)
        sorted_pred_lst = sorted(predicate_vocabulary)

        # We start enumerating entities and predicates starting from index 1, and leave 0 for the <UNKNOWN> symbol
        entity_index = {entity: idx for idx, entity in enumerate(sorted_ent_lst, start=1)}
        predicate_index = {predicate: idx for idx, predicate in enumerate(sorted_pred_lst, start=1)}

        return entity_index, predicate_index

    def facts_to_sequences(self, facts):
        """
        Transform each fact in facts as a sequence of symbol indexes.
        Only top 'nb_symbols' most frequent symbols will be taken into account.
        Returns a list of sequences.
        :param facts: lists of symbols.
        :return: list of individual sequences of indexes
        """
        return [indices for indices in self.facts_to_sequences_generator(facts)]

    def facts_to_sequences_generator(self, facts):
        """
        Transform each fact in facts as a pair (predicate_idx, argument_idxs),
        where predicate_idx is the index of the predicate, and argument_idxs is a list
        of indices associated to the arguments of the predicate.
        Yields individual pairs.
        :param facts: lists of facts.
        :return: yields individual (predicate_idx, argument_idxs) pairs.
        """
        for fact in facts:
            predicate_idx = self.predicate_to_index[fact.predicate_name]
            argument_idxs = [self.entity_to_index[arg] for arg in fact.argument_names]
            yield (predicate_idx, argument_idxs)
