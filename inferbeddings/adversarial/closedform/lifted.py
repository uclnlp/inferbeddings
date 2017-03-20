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

    def _distmult_loss(self, clause):
        head, body = clause.head, clause.body

        # At the moment, only simple rules as in "q(X, Y) :- p(X, Y)" are supported
        assert len(body) == 1
        body_atom = body[0]

        variable_names = {arg.name for arg in head.arguments} | {arg.name for arg in body_atom.arguments}
        assert len(variable_names) == 2

        # Indices of q and r, respectively
        head_predicate_idx = self.parser.predicate_to_index[head.predicate.name]
        body_predicate_idx = self.parser.predicate_to_index[body_atom.predicate.name]

        head_predicate_emb = tf.nn.embedding_lookup(self.predicate_embedding_layer, head_predicate_idx)
        body_predicate_emb = tf.nn.embedding_lookup(self.predicate_embedding_layer, body_predicate_idx)

        if self.is_unit_cube:
            loss = tf.reduce_sum(tf.nn.relu(body_predicate_emb - head_predicate_emb))
        else:
            loss = tf.reduce_max(tf.abs(body_predicate_emb - head_predicate_emb))
        return loss

    def __call__(self, clause):
        loss = None
        if self.model_class == BilinearDiagonalModel:
            # We are using DistMult
            loss = self._distmult_loss(clause)

        assert loss is not None
        return loss
