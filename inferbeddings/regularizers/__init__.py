# -*- coding: utf-8 -*-

from inferbeddings.regularizers.base import TransEEquivalentPredicateRegularizer,\
    DistMultEquivalentPredicateRegularizer,\
    ComplExEquivalentPredicateRegularizer
from inferbeddings.regularizers.util import clauses_to_equality_loss

__all__ = [
    'TransEEquivalentPredicateRegularizer',
    'DistMultEquivalentPredicateRegularizer',
    'ComplExEquivalentPredicateRegularizer',
    'clauses_to_equality_loss'
]
