# -*- coding: utf-8 -*-

from pyDatalog import pyDatalog

from inferbeddings.knowledgebase import Fact

import logging

logger = logging.getLogger(__name__)


def atom_to_str(atom, parser):
    # The predicate is represented by its index in pyDatalog, for avoiding syntax issues
    atom_predicate_idx = parser.predicate_to_index[atom.predicate.name]
    # atom_arg_0, atom_arg_1 are two variable names, like X and Y
    atom_arg_0, atom_arg_1 = atom.arguments[0], atom.arguments[1]
    # asserting P(X, p, Y)
    return 'p({}, {}, {})'.format(atom_arg_0, atom_predicate_idx, atom_arg_1)


def clause_to_str(clause, parser):
    head, body = clause.head, clause.body

    body_str = ' & '.join([atom_to_str(a, parser) for a in body])
    return '{} <= {}'.format(atom_to_str(head, parser), body_str)


def materialize(facts, clauses, parser):
    logger.info('Asserting facts ..')
    for f in facts:
        # Each fact is asserted using the index of the subject, predicate and object for avoiding syntax issues
        s_idx = parser.entity_to_index[f.argument_names[0]]
        p_idx = parser.predicate_to_index[f.predicate_name]
        o_idx = parser.entity_to_index[f.argument_names[1]]
        # Asserting p(S, P, O)
        pyDatalog.assert_fact('p', s_idx, p_idx, o_idx)

    rules_str = '\n'.join([clause_to_str(clause, parser) for clause in clauses])
    pyDatalog.load(rules_str)

    # Asking for all P(s, p, o) triples which hold true in the Knowledge Graph
    logger.info('Querying triples ..')
    _ans = pyDatalog.ask('p(S, P, O)')

    index_to_predicate = {idx: p for p, idx in parser.predicate_to_index.items()}
    index_to_entity = {idx: e for e, idx in parser.entity_to_index.items()}

    # Generating a list of inferred facts by replacing each entity and predicate index with their corresponding symbols
    inferred_facts = [Fact(index_to_predicate[p], [index_to_entity[s], index_to_entity[o]])
                      for (s, p, o) in sorted(_ans.answers)]
    return inferred_facts
