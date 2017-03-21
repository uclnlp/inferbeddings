# -*- coding: utf-8 -*-

import tensorflow as tf

from inferbeddings.models import TranslatingModel, BilinearDiagonalModel, ComplexModel
from inferbeddings.models import similarities

import logging

logger = logging.getLogger(__name__)


class ClosedFormLifted:
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

        # At te moment we only support "r(X, Y) :- b(X, Y)" rules, and not "r(X, Y) :- b(Y, X)"
        assert head.arguments[0].name == body_atom.arguments[0].name
        assert head.arguments[1].name == body_atom.arguments[1].name

        # We only support TransE in its L2 squared distance formulation
        assert self.model_parameters['similarity_function'] == similarities.l2_sqr

        # Indices of q and r, respectively
        r_idx, b_idx = self._to_idx(head.predicate.name), self._to_idx(body_atom.predicate.name)

        r = tf.nn.embedding_lookup(self.predicate_embedding_layer, r_idx)
        b = tf.nn.embedding_lookup(self.predicate_embedding_layer, b_idx)

        prefix = tf.reduce_sum(tf.square(r)) - tf.reduce_sum(tf.square(b))
        if self.is_unit_cube:
            loss = tf.nn.relu(prefix + 2 * tf.reduce_sum(tf.abs(r - b)))
        else:
            loss = tf.nn.relu(prefix + 4 * tf.sqrt(tf.reduce_sum(tf.square(r - b))))
        return loss

    def _bilinear_diagonal_loss(self, clause):
        head, body = clause.head, clause.body

        # At the moment, only simple rules as in "r(X, Y) :- b(X, Y)" are supported
        assert len(body) == 1
        body_atom = body[0]

        variable_names = {arg.name for arg in head.arguments} | {arg.name for arg in body_atom.arguments}
        assert len(variable_names) == 2

        # Indices of q and r, respectively
        r_idx, b_idx = self._to_idx(head.predicate.name), self._to_idx(body_atom.predicate.name)

        r = tf.nn.embedding_lookup(self.predicate_embedding_layer, r_idx)
        b = tf.nn.embedding_lookup(self.predicate_embedding_layer, b_idx)

        if self.is_unit_cube:
            loss = tf.reduce_sum(tf.nn.relu(b - r))
        else:
            loss = tf.reduce_max(tf.abs(b - r))
        return loss

    def _complex_loss(self, clause):
        head, body = clause.head, clause.body

        # At the moment, only simple rules as in "r(X, Y) :- b(X, Y)" are supported
        assert len(body) == 1
        body_atom = body[0]

        variable_names = {arg.name for arg in head.arguments} | {arg.name for arg in body_atom.arguments}
        assert len(variable_names) == 2

        # Indices of q and r, respectively
        r_idx, b_idx = self._to_idx(head.predicate.name), self._to_idx(body_atom.predicate.name)

        r = tf.nn.embedding_lookup(self.predicate_embedding_layer, r_idx)
        b = tf.nn.embedding_lookup(self.predicate_embedding_layer, b_idx)

        n = r.get_shape()[-1].value
        r_re, r_im = r[:n // 2], r[n // 2:]
        b_re, b_im = b[:n // 2], b[n // 2:]

        if self.is_unit_cube:
            loss = None
        else:
            loss = tf.reduce_max(tf.sqrt(tf.square(b_re - r_re) + tf.square(b_im - r_im)))
        return loss

    def __call__(self, clause):
        loss = None
        if self.model_class == BilinearDiagonalModel:
            # We are using DistMult
            loss = self._bilinear_diagonal_loss(clause)
        elif self.model_class == TranslatingModel:
            # We are using DistMult
            loss = self._translating_loss(clause)

        assert loss is not None
        return loss
