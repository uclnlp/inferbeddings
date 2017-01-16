# -*- coding: utf-8 -*-

import logging

logger = logging.getLogger(__name__)


class GroundLoss:
    def __init__(self, clauses, entity_to_index, predicate_to_index, scoring_function):
        self.clauses = clauses
        self.entity_to_index = entity_to_index
        self.predicate_to_index = predicate_to_index
        self.scoring_function = scoring_function

    @staticmethod
    def get_variable_names(clause):
        """
        Utility method to retrieve the variables contained in a clause.
        :param clause: Clause.
        :return: Set of variable names.
        """
        head, body = clause.head, clause.body
        variable_names = {argument.name for argument in head.arguments}
        for body_atom in body:
            variable_names |= {argument.name for argument in body_atom.arguments}
        return variable_names

    def __entity_to_idx(self, entity):
        return self.entity_to_index[entity] if isinstance(entity, str) else entity

    def __predicate_to_idx(self, predicate):
        return self.predicate_to_index[predicate] if isinstance(predicate, str) else predicate

    def _score_atom(self, atom, feed_dict):
        arg1_name, arg2_name, predicate_name = atom.arguments[0].name, atom.arguments[1].name, atom.predicate.name
        arg1_value, arg2_value = feed_dict[arg1_name], feed_dict[arg2_name]

        s_idx, o_idx = self.__entity_to_idx(arg1_value), self.__entity_to_idx(arg2_value)
        p_idx = self.__predicate_to_idx(predicate_name)

        score_value = self.scoring_function([[[p_idx]], [[s_idx, o_idx]]])
        return score_value

    def _score_conjunction(self, atoms, feed_dict):
        atom_scores = [self._score_atom(atom, feed_dict) for atom in atoms]
        return min(atom_scores)

    def zero_one_error(self, clause, feed_dict):
        """
        Compute the 0-1 loss of a clause w.r.t. of a variable assignment feed_dict
        :param clause: Clause.
        :param feed_dict: Variable assignment: {variable_name: entity}
        :return: Value in {0, 1}
        """
        head, body = clause.head, clause.body
        score_head = self._score_atom(head, feed_dict)
        score_body = self._score_conjunction(body, feed_dict)
        return int(not score_body <= score_head)
