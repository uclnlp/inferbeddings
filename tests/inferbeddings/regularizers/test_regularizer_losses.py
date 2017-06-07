# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from inferbeddings.knowledgebase import Fact, KnowledgeBaseParser
from inferbeddings.parse import parse_clause
from inferbeddings.regularizers.util import clauses_to_equality_loss

import pytest


@pytest.mark.light
def test_losses():
    triples = [
        ('e1', 'p', 'e2'),
        ('e2', 'q', 'e3'),
        ('e1', 'r', 'e2'),
        ('e2', 's', 'e3')
    ]

    def fact(s, p, o):
        return Fact(predicate_name=p, argument_names=[s, o])

    facts = [fact(s, p, o) for s, p, o in triples]
    parser = KnowledgeBaseParser(facts)

    nb_predicates = len(parser.predicate_vocabulary)
    predicate_embedding_size = 100
    predicate_embedding_layer = tf.get_variable('predicates',
                                                shape=[nb_predicates + 1, predicate_embedding_size],
                                                initializer=tf.contrib.layers.xavier_initializer())

    clauses = [parse_clause('p(X, Y) :- q(Y, X)'), parse_clause('r(X, Y) :- s(X, Y)')]
    loss = clauses_to_equality_loss('TransE', clauses, 'l2_sqr',
                                    predicate_embedding_layer,
                                    parser.predicate_to_index)

    for i in range(32):
        optimizer = tf.train.AdagradOptimizer(0.1)
        minimization_step = optimizer.minimize(loss, var_list=[predicate_embedding_layer])

        init_op = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init_op)

            for j in range(32):
                session.run([minimization_step])

                loss_value = session.run([loss])[0]

                p_idx, q_idx = parser.predicate_to_index['p'], parser.predicate_to_index['q']
                r_idx, s_idx = parser.predicate_to_index['r'], parser.predicate_to_index['s']

                predicate_embedding_layer_value = session.run([predicate_embedding_layer])[0]

                p_value, q_value = predicate_embedding_layer_value[p_idx, :], predicate_embedding_layer_value[q_idx, :]
                r_value, s_value = predicate_embedding_layer_value[r_idx, :], predicate_embedding_layer_value[s_idx, :]

                estimated_loss_value = np.square(p_value + q_value).sum() + np.square(r_value - s_value).sum()

                assert loss_value > 0
                assert estimated_loss_value > 0
                np.testing.assert_allclose(loss_value, estimated_loss_value, 4)

    tf.reset_default_graph()

if __name__ == '__main__':
    pytest.main([__file__])
