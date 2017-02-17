# -*- coding: utf-8 -*-

from pyDatalog import pyDatalog

from inferbeddings.knowledgebase import Fact

import logging

logger = logging.getLogger(__name__)


def atom_to_str(atom, predicate_to_idx):
    atom_predicate_idx = predicate_to_idx[atom.predicate.name]
    atom_arg_0, atom_arg_1 = atom.arguments[0], atom.arguments[1]
    return 'p({}, {}, {})'.format(atom_arg_0, atom_predicate_idx, atom_arg_1)


def clause_to_str(clause, predicate_to_idx):
    head, body = clause.head, clause.body
    body_str = ' & '.join([atom_to_str(a, predicate_to_idx) for a in body])
    return '{} <= {}'.format(atom_to_str(head, predicate_to_idx), body_str)


def materialize(facts, clauses, predicate_to_idx):
    logger.info('Asserting facts ..')
    for f in facts:
        pyDatalog.assert_fact('p', f.argument_names[0], f.predicate_name, f.argument_names[1])

    rules_str = '\n'.join([clause_to_str(clause, predicate_to_idx) for clause in clauses])
    pyDatalog.load(rules_str)

    logger.info('Querying triples ..')
    _ans = pyDatalog.ask('p(S, P, O)')
    inferred_facts = [Fact(p, [s, o]) for (s, p, o) in sorted(_ans.answers)]

    return inferred_facts
