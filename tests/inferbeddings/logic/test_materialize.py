# -*- coding: utf-8 -*-

import pytest

from inferbeddings.knowledgebase import Fact
from inferbeddings.parse import parse_clause

from inferbeddings.logic import materialize


def test_materialize():
    initial_facts = [Fact(1, [idx, idx + 1]) for idx in range(64)]

    predicate_to_idx = {'q': 1}

    clause_str = 'q(X, Z) :- q(X, Y), q(Y, Z)'
    clause = parse_clause(clause_str)

    inferred_facts = materialize(initial_facts, [clause], predicate_to_idx)
    inferred_triples = [(f.argument_names[0], f.predicate_name, f.argument_names[1]) for f in inferred_facts]

    entities = {s for (s, _, _) in inferred_triples} | {o for (_, _, o) in inferred_triples}

    for e1 in entities:
        for e2 in entities:
            if e1 < e2:
                assert (e1, 1, e2) in inferred_triples
            else:
                assert (e1, 1, e2) not in inferred_triples

if __name__ == '__main__':
    pytest.main([__file__])
