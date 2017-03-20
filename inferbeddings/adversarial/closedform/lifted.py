# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from inferbeddings.models import TranslatingModel, BilinearDiagonalModel, ComplexModel

import logging

logger = logging.getLogger(__name__)


class ClosedForm:
    def __init__(self, parser,
                 predicate_embedding_layer,
                 model_class, model_parameters,
                 is_unit_cube):
        self.parser = parser
        self.predicate_embedding_layer = predicate_embedding_layer
        self.model_class, self.model_parameters = model_class, model_parameters
        self.is_unit_cube = is_unit_cube

    def _to_idx(self, predicate_name):
        return self.parser.predicate_to_index[predicate_name]

    def _translating_loss(self, clause):
        head, body = clause.head, clause.body

        # At the moment, only simple rules as in "r(X, Y) :- b(X, Y)" are supported
        assert len(body) == 1
        body_atom = body[0]

        variable_names = {arg.name for arg in head.arguments} | {arg.name for arg in body_atom.arguments}
        assert len(variable_names) == 2

        # Indices of q and r, respectively
        r_idx, b_idx = self._to_idx[head.predicate.name], self._to_idx[body_atom.predicate.name]

        r = tf.nn.embedding_lookup(self.predicate_embedding_layer, r_idx)
        b = tf.nn.embedding_lookup(self.predicate_embedding_layer, b_idx)

        prefix = tf.reduce_sum(tf.square(b)) - tf.reduce_sum(tf.square(r))
        if self.is_unit_cube:
            loss = tf.nn.relu(prefix + 2 * tf.reduce_sum(tf.abs(b - r)))
        else:
            loss = tf.nn.relu(prefix + 4 * tf.sqrt(tf.reduce_sum(tf.square(b - r))))
        return loss

    def _bilinear_diagonal_loss(self, clause):
        head, body = clause.head, clause.body

        # At the moment, only simple rules as in "r(X, Y) :- b(X, Y)" are supported
        assert len(body) == 1
        body_atom = body[0]

        variable_names = {arg.name for arg in head.arguments} | {arg.name for arg in body_atom.arguments}
        assert len(variable_names) == 2

        # Indices of q and r, respectively
        r_idx, b_idx = self._to_idx[head.predicate.name], self._to_idx[body_atom.predicate.name]

        r = tf.nn.embedding_lookup(self.predicate_embedding_layer, r_idx)
        b = tf.nn.embedding_lookup(self.predicate_embedding_layer, b_idx)

        if self.is_unit_cube:
            loss = tf.reduce_sum(tf.nn.relu(b - r))
        else:
            loss = tf.reduce_max(tf.abs(b - r))
        return loss

    def __call__(self, clause):
        loss = None
        if self.model_class == BilinearDiagonalModel:
            # We are using DistMult
            loss = self._bilinear_diagonal_loss(clause)

        assert loss is not None
        return loss
