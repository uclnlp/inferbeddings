# -*- coding: utf-8 -*-

import numpy as np

from inferbeddings.models import BilinearDiagonalModel

import logging

logger = logging.getLogger(__name__)


class ClosedForm:
    def __init__(self, parser,
                 entity_embeddings, predicate_embeddings,
                 model_class, model_parameters):
        self.parser = parser
        self.entity_embeddings, self.predicate_embeddings = entity_embeddings, predicate_embeddings
        self.model_class, self.model_parameters = model_class, model_parameters

        self.entity_embedding_size = self.entity_embeddings.shape[0]

    def _distmult_unitbox(self, clause):
        head, body = clause.head, clause.body

        # At the moment, only simple rules as in "q(X, Y) :- p(X, Y)" are supported
        assert len(body) == 1
        body_atom = body[0]

        variable_names = {argument.name for argument in head.arguments}

        head_predicate_idx = self.parser.predicate_to_index[head.predicate.name]
        body_predicate_idx = self.parser.predicate_to_index[body_atom.predicate.name]

        head_predicate_emb = self.predicate_embeddings[head_predicate_idx, :]
        body_predicate_emb = self.predicate_embeddings[body_predicate_idx, :]

        optimal_emb = (head_predicate_emb >= body_predicate_emb).astype(np.float32)

        return {var_name: optimal_emb for var_name in variable_names}

    def _distmult_unitball(self, clause):
        head, body = clause.head, clause.body

        # At the moment, only simple rules as in "q(X, Y) :- p(X, Y)" are supported
        assert len(body) == 1
        body_atom = body[0]

        variable_names = {argument.name for argument in head.arguments}

        head_predicate_idx = self.parser.predicate_to_index[head.predicate.name]
        body_predicate_idx = self.parser.predicate_to_index[body_atom.predicate.name]

        head_predicate_emb = self.predicate_embeddings[head_predicate_idx, :]
        body_predicate_emb = self.predicate_embeddings[body_predicate_idx, :]

        j = np.square(body_predicate_emb - head_predicate_emb).argmax(axis=0)

        opt_emb_X, opt_emb_Y = np.zeros(self.entity_embedding_size), np.zeros(self.entity_embedding_size)
        opt_emb_X[j], opt_emb_Y[j] = 1, 1 if (opt_emb_X[j] > opt_emb_Y[j]) else -1

        variable_names_lst = list(variable_names)
        return {variable_names_lst[0]: opt_emb_X, variable_names_lst[1]: opt_emb_Y}

    def __call__(self, clause):
        if self.model_class == BilinearDiagonalModel:
            # We are using DistMult

