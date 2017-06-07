# -*- coding: utf-8 -*-

import pytest

from inferbeddings.knowledgebase import Fact, KnowledgeBaseParser
from inferbeddings.parse import parse_clause

from inferbeddings.logic import materialize


@pytest.mark.light
def test_materialize():
    initial_facts = [Fact('q', ['{}'.format(idx), '{}'.format(idx + 1)]) for idx in range(64)]
    parser = KnowledgeBaseParser(initial_facts)
    parser.predicate_to_index['p'] = 2

    clauses = [
        parse_clause('q(X, Z) :- q(X, Y), q(Y, Z)'),
        parse_clause('p(X, Y) :- q(X, Y)')
    ]

    inferred_facts = materialize(initial_facts, clauses, parser)
    inferred_triples = [(f.argument_names[0], f.predicate_name, f.argument_names[1]) for f in inferred_facts]

    entities = {s for (s, _, _) in inferred_triples} | {o for (_, _, o) in inferred_triples}

    for e1 in entities:
        for e2 in entities:
            if int(e1) < int(e2):
                assert (str(e1), 'q', str(e2)) in inferred_triples
                assert (str(e1), 'p', str(e2)) in inferred_triples
                print('+')
            else:
                assert (str(e1), 'q', str(e2)) not in inferred_triples
                assert (str(e1), 'p', str(e2)) not in inferred_triples
                print('-')

if __name__ == '__main__':
    pytest.main([__file__])
