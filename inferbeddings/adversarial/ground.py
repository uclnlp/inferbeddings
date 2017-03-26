# -*- coding: utf-8 -*-

import numpy as np
import logging

logger = logging.getLogger(__name__)


class GroundLoss:
    def __init__(self, clauses, parser, scoring_function, tolerance=0.0):
        self.clauses = clauses
        self.parser = parser
        self.scoring_function = scoring_function
        self.tolerance = tolerance

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

    @staticmethod
    def __tuple_to_mapping(variables, _tuple):
        return {var_name: var_value for var_name, var_value in zip(variables, _tuple)}

    @staticmethod
    def sample_mappings(variables, entities, sample_size=1024, seed=None):
        """
        Sample a sample_size set of {variable:entity} mappings without replacement.
        :param variables: List of variables, e.g. ['X', 'Y', 'Z']
        :param entities: List of entities, e.g. [1, 2, 3]
        :param sample_size: sample size.
        :param seed: Random seed.
        :return: List of randomly sampled {variable:entity} mappings.
        """
        rs = np.random.RandomState(0 if seed is None else seed)
        nb_entities, nb_variables, np_entities = len(entities), len(variables), np.array(entities)
        tuple_set, sample_size = set(), min(nb_entities ** nb_variables, sample_size)

        while len(tuple_set) < sample_size:
            tuple_set |= {tuple(value for value in np_entities[rs.choice(nb_entities, nb_variables)])}

        return [GroundLoss.__tuple_to_mapping(variables, _tuple) for _tuple in tuple_set]

    def __entity_to_idx(self, entity):
        return self.parser.entity_to_index[entity] if isinstance(entity, str) else entity

    def __predicate_to_idx(self, predicate):
        return self.parser.predicate_to_index[predicate] if isinstance(predicate, str) else predicate

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

    def zero_one_errors(self, clause, feed_dicts):
        return sum([self.zero_one_error(clause, feed_dict) for feed_dict in feed_dicts])

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
        return int(not ((score_body - self.tolerance) <= score_head))

    def continuous_errors(self, clause, feed_dicts):
        return sum([self.continuous_error(clause, feed_dict) for feed_dict in feed_dicts])

    def continuous_error(self, clause, feed_dict):
        """
        Compute the violation error of a clause w.r.t. of a variable assignment feed_dict
        :param clause: Clause.
        :param feed_dict: Variable assignment: {variable_name: entity}
        :return: Continuous value
        """
        head, body = clause.head, clause.body
        score_head = self._score_atom(head, feed_dict)
        score_body = self._score_conjunction(body, feed_dict)
        return score_body - score_head
