# -*- coding: utf-8 -*-

import numpy as np

from inferbeddings.models import BilinearDiagonalModel, ComplexModel

from inferbeddings.adversarial.closedform.util import score_complex

import logging

logger = logging.getLogger(__name__)


class ClosedForm:
    def __init__(self, parser,
                 entity_embeddings, predicate_embeddings,
                 model_class, model_parameters,
                 is_unit_cube):
        self.parser = parser
        self.entity_embeddings, self.predicate_embeddings = entity_embeddings, predicate_embeddings
        self.model_class, self.model_parameters = model_class, model_parameters
        self.is_unit_cube = is_unit_cube

        self.entity_embedding_size = self.entity_embeddings.shape[0]

    def _complex_unit_cube(self, clause):
        head, body = clause.head, clause.body

        # At the moment, only simple rules as in "q(X, Y) :- p(X, Y)" are supported
        assert len(body) == 1
        body_atom = body[0]

        assert head.arguments[0].name != head.arguments[1].name
        assert body_atom.arguments[0].name != body_atom.arguments[1].name

        variable_names = {arg.name for arg in head.arguments} | {arg.name for arg in body_atom.arguments}
        assert len(variable_names) == 2

        head_predicate_idx = self.parser.predicate_to_index[head.predicate.name]
        body_predicate_idx = self.parser.predicate_to_index[body_atom.predicate.name]

        head_predicate_emb = self.predicate_embeddings[head_predicate_idx, :]
        body_predicate_emb = self.predicate_embeddings[body_predicate_idx, :]

        n = head_predicate_emb.shape[0]
        opt_emb_X, opt_emb_Y = np.zeros(n), np.zeros(n)

        for j in range(n // 2):
            candidates = [
                (1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 0.0, 1.0), (1.0, 1.0, 1.0, 0.0),
                (0.0, 1.0, 1.0, 0.0), (1.0, 0.0, 0.0, 1.0)
            ]
            highest_loss_value, best_candidate = None, None
            for (sR_j, oR_j, sI_j, oI_j) in candidates:
                _opt_emb_s, _opt_emb_o = np.copy(opt_emb_X), np.copy(opt_emb_Y)
                _opt_emb_s[j], _opt_emb_o[j] = sR_j, oR_j
                _opt_emb_s[(n // 2) + j], _opt_emb_o[(n // 2) + j] = sI_j, oI_j
                loss_value = None
                if head.arguments[0].name == body_atom.arguments[0].name and \
                        head.arguments[1].name == body_atom.arguments[1].name:
                    loss_value = score_complex(_opt_emb_s, body_predicate_emb, _opt_emb_o) - \
                                 score_complex(_opt_emb_s, head_predicate_emb, _opt_emb_o)
                elif head.arguments[0].name == body_atom.arguments[1].name and \
                        head.arguments[1].name == body_atom.arguments[0].name:
                    loss_value = score_complex(_opt_emb_o, body_predicate_emb, _opt_emb_s) -\
                                 score_complex(_opt_emb_s, head_predicate_emb, _opt_emb_o)
                assert loss_value is not None
                if highest_loss_value is None or loss_value > highest_loss_value:
                    highest_loss_value = loss_value
                    best_candidate = (sR_j, oR_j, sI_j, oI_j)
            opt_emb_X[j], opt_emb_Y[j] = best_candidate[0], best_candidate[1]
            opt_emb_X[(n // 2) + j], opt_emb_Y[(n // 2) + j] = best_candidate[2], best_candidate[3]

        variable_names_lst = list(variable_names)
        return {variable_names_lst[0]: opt_emb_X, variable_names_lst[1]: opt_emb_Y}

    def _distmult_unit_cube(self, clause):
        head, body = clause.head, clause.body

        # At the moment, only simple rules as in "q(X, Y) :- p(X, Y)" are supported
        assert len(body) == 1
        body_atom = body[0]

        variable_names = {arg.name for arg in head.arguments} | {arg.name for arg in body_atom.arguments}
        assert len(variable_names) == 2

        head_predicate_idx = self.parser.predicate_to_index[head.predicate.name]
        body_predicate_idx = self.parser.predicate_to_index[body_atom.predicate.name]

        head_predicate_emb = self.predicate_embeddings[head_predicate_idx, :]
        body_predicate_emb = self.predicate_embeddings[body_predicate_idx, :]

        optimal_emb = (head_predicate_emb >= body_predicate_emb).astype(np.float32)

        return {var_name: optimal_emb for var_name in variable_names}

    def _distmult_unit_sphere(self, clause):
        head, body = clause.head, clause.body

        # At the moment, only simple rules as in "q(X, Y) :- p(X, Y)" are supported
        assert len(body) == 1
        body_atom = body[0]

        variable_names = {arg.name for arg in head.arguments} | {arg.name for arg in body_atom.arguments}
        assert len(variable_names) == 2

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
        opt_adv_emb = None
        if self.model_class == BilinearDiagonalModel:
            # We are using DistMult
            if self.is_unit_cube:
                opt_adv_emb = self._distmult_unit_cube(clause)
            else:
                opt_adv_emb = self._distmult_unit_sphere(clause)
        elif self.model_class == ComplexModel:
            # We are using ComplEx
            if self.is_unit_cube:
                opt_adv_emb = self._complex_unit_cube(clause)

        assert opt_adv_emb is not None
        return opt_adv_emb

