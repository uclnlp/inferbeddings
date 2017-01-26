# -*- coding: utf-8 -*-

import pytest

import inferbeddings.parse.clauses as clauses


def test_parse_clauses_one():
    clause_str = 'p(x, y) :- p(x, z), q(z, a), r(a, y)'

    parsed = clauses.grammar.parse(clause_str)
    clause = clauses.ClauseVisitor().visit(parsed)

    assert isinstance(clause, clauses.Clause)

    assert isinstance(clause.head, clauses.Atom)
    assert isinstance(clause.body, tuple)

    assert isinstance(clause.head.predicate, clauses.Predicate)
    assert isinstance(clause.head.arguments, tuple)
    assert isinstance(clause.head.negated, bool)


def test_parse_clauses_two():
    clause_str = '"P"(x, y) :- p(x, z), q(z, a), "R"(a, y)'

    parsed = clauses.grammar.parse(clause_str)
    clause = clauses.ClauseVisitor().visit(parsed)

    assert isinstance(clause, clauses.Clause)

    assert isinstance(clause.head, clauses.Atom)
    assert isinstance(clause.head.predicate.name, str)
    assert isinstance(clause.body, tuple)

    assert isinstance(clause.head.predicate, clauses.Predicate)
    assert isinstance(clause.head.arguments, tuple)
    assert isinstance(clause.head.negated, bool)


if __name__ == '__main__':
    pytest.main([__file__])
