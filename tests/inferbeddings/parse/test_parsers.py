# -*- coding: utf-8 -*-

import pytest

import inferbeddings.parse.clauses as clauses


@pytest.mark.light
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
    assert clause.weight == 1.0


@pytest.mark.light
def test_parse_atom_clause():
    clause_str = 'p(X, y)'

    parsed = clauses.grammar.parse(clause_str)
    clause = clauses.ClauseVisitor().visit(parsed)

    assert isinstance(clause, clauses.Clause)

    assert isinstance(clause.head, clauses.Atom)
    assert len(clause.body) == 0
    assert clause.head.predicate.name == "p"
    assert isinstance(clause.head.arguments[0], clauses.Variable)
    assert isinstance(clause.head.arguments[1], clauses.Constant)
    assert clause.head.arguments[1].name == "y"
    assert clause.weight == 1.0


@pytest.mark.light
def test_parse_weighted_atom_clause():
    clause_str = 'p(X, y) < -1.2 >'
    parsed = clauses.grammar.parse(clause_str)
    clause = clauses.ClauseVisitor().visit(parsed)
    assert clause.weight == -1.2


@pytest.mark.light
def test_parse_weighted_arity_2_clause():
    clause_str = 'p(X, y) :- r(X,Z), q(X) < 1.2 >'
    parsed = clauses.grammar.parse(clause_str)
    clause = clauses.ClauseVisitor().visit(parsed)
    assert clause.weight == 1.2


@pytest.mark.light
def test_parse_learnable_weight_arity_2_clause():
    clause_str = 'p(X, y) :- r(X,Z), q(X) < ? >'
    parsed = clauses.grammar.parse(clause_str)
    clause = clauses.ClauseVisitor().visit(parsed)
    assert clause.weight is None


@pytest.mark.light
def test_parse_learnable_weight_atom_clause():
    clause_str = 'p(X, y) < ? >'
    parsed = clauses.grammar.parse(clause_str)
    clause = clauses.ClauseVisitor().visit(parsed)
    assert clause.weight is None


@pytest.mark.light
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
    assert clause.weight == 1.0


if __name__ == '__main__':
    pytest.main([__file__])
