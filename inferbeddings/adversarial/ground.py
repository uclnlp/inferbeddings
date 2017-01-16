# -*- coding: utf-8 -*-

import logging

logger = logging.getLogger(__name__)


class GroundLoss:
    def __init__(self, clauses, entity_to_index, predicate_to_index, scoring_function):
        self.clauses = clauses

        self.entity_to_index = entity_to_index
        self.predicate_to_index = predicate_to_index

        self.scoring_function = scoring_function

    def __entity_to_idx(self, entity):
        return self.entity_to_index[entity] if isinstance(entity, str) else entity

    def __predicate_to_idx(self, predicate):
        return self.predicate_to_index[predicate] if isinstance(predicate, str) else predicate

    def _score_atom(self, atom, feed_dict):
        arg1_name, arg2_name, predicate_name = atom.arguments[0].name, atom.arguments[1].name, atom.predicate.name
        arg1_value, arg2_value = feed_dict[arg1_name], feed_dict[arg2_name]
        score_value = self.scoring_function(self.__entity_to_idx(arg1_value),
                                            self.__predicate_to_idx(predicate_name),
                                            self.__entity_to_idx(arg2_value))
        return score_value

    def _score_conjunction(self, atoms, feed_dict):
        atom_scores = [self._score_atom(atom, feed_dict) for atom in atoms]
        return min(atom_scores)

    def _error_clause(self, clause, feed_dict):
        head, body = clause.head, clause.body
        score_head = self._score_atom(head, feed_dict)
        score_body = self._score_conjunction(body, feed_dict)
        return int(not score_body <= score_head)

    def error(self, clauses, feed_dict):
        return sum([self._error_clause(clause, feed_dict) for clause in clauses])
