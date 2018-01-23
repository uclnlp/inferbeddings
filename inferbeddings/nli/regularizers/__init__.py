# -*- coding: utf-8 -*-

from inferbeddings.nli.regularizers.base import contradiction_symmetry_l1
from inferbeddings.nli.regularizers.base import contradiction_symmetry_l2
from inferbeddings.nli.regularizers.base import contradiction_kullback_leibler
from inferbeddings.nli.regularizers.base import contradiction_jensen_shannon

from inferbeddings.nli.regularizers.base import contradiction_acl
from inferbeddings.nli.regularizers.base import entailment_acl
from inferbeddings.nli.regularizers.base import neutral_acl

from inferbeddings.nli.regularizers.base import entailment_reflexive_acl

__all__ = [
    'contradiction_symmetry_l1',
    'contradiction_symmetry_l2',
    'contradiction_kullback_leibler',
    'contradiction_jensen_shannon',

    'contradiction_acl',
    'entailment_acl',
    'neutral_acl',

    'entailment_reflexive_acl'
]
