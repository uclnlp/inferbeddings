# -*- coding: utf-8 -*-

from parse import clauses


def parse_clause(text):
    parsed = clauses.grammar.parse(text)
    return clauses.ClauseVisitor().visit(parsed)
