# -*- coding: utf-8 -*-

import tensorflow as tf

from inferbeddings.regularizers import TransEEquivalentPredicateRegularizer
from inferbeddings.regularizers import DistMultEquivalentPredicateRegularizer
from inferbeddings.regularizers import ComplExEquivalentPredicateRegularizer
from inferbeddings.regularizers import BilinearEquivalentPredicateRegularizer

import logging

logger = logging.getLogger(__name__)


def _model_name_to_regularizer_class(model_name):
    regularizer_class = None
    if model_name == 'TransE':
        regularizer_class = TransEEquivalentPredicateRegularizer
    elif model_name == 'DistMult':
        regularizer_class = DistMultEquivalentPredicateRegularizer
    elif model_name == 'ComplEx':
        regularizer_class = ComplExEquivalentPredicateRegularizer
    elif model_name in {'RESCAL', 'Bilinear'}:
        regularizer_class = BilinearEquivalentPredicateRegularizer
    return regularizer_class


def clauses_to_equality_loss(model_name,
                             clauses,
                             similarity_name,
                             predicate_embedding_layer,
                             predicate_to_index,
                             entity_embedding_size):
    loss = 0.0

    regularizer_class = _model_name_to_regularizer_class(model_name)
    assert regularizer_class is not None

    _added_clauses = set()

    for clause in clauses:
        head, body = clause.head, clause.body
        assert len(body) == 1
        body_atom = body[0]

        head_arg1, head_arg2 = head.arguments[0].name, head.arguments[1].name
        body_arg1, body_arg2 = body_atom.arguments[0].name, body_atom.arguments[1].name

        is_inverse = None
        if head_arg1 == body_arg1 and head_arg2 == body_arg2:
            is_inverse = False
        elif head_arg2 == body_arg1 and head_arg1 == body_arg2:
            is_inverse = True
        assert is_inverse is not None

        assert head.predicate.name in predicate_to_index
        assert body_atom.predicate.name in predicate_to_index

        head_predicate_idx = predicate_to_index[head.predicate.name]
        body_predicate_idx = predicate_to_index[body_atom.predicate.name]

        _head_tuple = (head_predicate_idx, (head_arg1, head_arg2))
        _body_tuple = (body_predicate_idx, (body_arg1, body_arg2))

        if (_head_tuple, _body_tuple) not in _added_clauses:
            _added_clauses |= {(_head_tuple, _body_tuple), (_body_tuple, _head_tuple)}

            head_predicate_embedding = tf.nn.embedding_lookup(predicate_embedding_layer, head_predicate_idx)
            body_predicate_embedding = tf.nn.embedding_lookup(predicate_embedding_layer, body_predicate_idx)

            regularizer = regularizer_class(x1=head_predicate_embedding, x2=body_predicate_embedding,
                                            is_inverse=is_inverse, similarity_name=similarity_name,
                                            entity_embedding_size=entity_embedding_size)
            loss += regularizer()
    return loss
