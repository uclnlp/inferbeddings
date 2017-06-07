# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from inferbeddings.knowledgebase import Fact, KnowledgeBaseParser
from inferbeddings.parse import parse_clause
from inferbeddings.models import base as models
from inferbeddings.models import similarities

from inferbeddings.adversarial import Adversarial

import pytest


@pytest.mark.light
def test_adversarial():
    for _ in range(32):
        _test_adversarial()


def _test_adversarial():
    triples = [
        ('john', 'friendOf', 'mark'),
        ('mark', 'friendOf', 'aleksi'),
        ('mark', 'friendOf', 'dazdrasmygda')
    ]

    def fact(s, p, o):
        return Fact(predicate_name=p, argument_names=[s, o])

    facts = [fact(s, p, o) for s, p, o in triples]
    parser = KnowledgeBaseParser(facts)
    clauses = [parse_clause('friendOf(X, Y) :- friendOf(Y, X)')]

    nb_entities = len(parser.entity_vocabulary)
    nb_predicates = len(parser.predicate_vocabulary)

    entity_embedding_size = 100
    predicate_embedding_size = 100

    entity_embedding_layer = tf.get_variable('entities', shape=[nb_entities + 1, entity_embedding_size],
                                             initializer=tf.contrib.layers.xavier_initializer())

    predicate_embedding_layer = tf.get_variable('predicates',
                                                shape=[nb_predicates + 1, predicate_embedding_size],
                                                initializer=tf.contrib.layers.xavier_initializer())

    model_class = models.get_function('TransE')

    similarity_function = similarities.get_function('l1')
    model_parameters = dict(similarity_function=similarity_function)

    batch_size = 1000

    adversarial = Adversarial(clauses=clauses,
                              parser=parser,
                              entity_embedding_layer=entity_embedding_layer,
                              predicate_embedding_layer=predicate_embedding_layer,
                              model_class=model_class,
                              model_parameters=model_parameters,
                              batch_size=batch_size)

    init_op = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init_op)
        assert len(adversarial.parameters) == 2
        for violating_embeddings in adversarial.parameters:
            shape = session.run(tf.shape(violating_embeddings))
            assert (shape == (batch_size, entity_embedding_size)).all()

        loss_value = session.run(adversarial.loss)
        errors_value = session.run(adversarial.errors)

        var1 = adversarial.parameters[0]
        var2 = adversarial.parameters[1]

        X_values = session.run(var1 if "X" in var1.name else var2)
        Y_values = session.run(var2 if "Y" in var2.name else var1)

        p_value = session.run(tf.nn.embedding_lookup(predicate_embedding_layer, 1))

        assert np.array(X_values.shape == (batch_size, entity_embedding_size)).all()
        assert np.array(Y_values.shape == (batch_size, entity_embedding_size)).all()
        assert np.array(p_value.shape == (predicate_embedding_size,))

        head_scores = - np.sum(np.abs((X_values + p_value) - Y_values), axis=1)
        body_scores = - np.sum(np.abs((Y_values + p_value) - X_values), axis=1)

        assert int(errors_value) == np.sum((head_scores < body_scores).astype(int))

        linear_losses = body_scores - head_scores
        np_loss_values = np.sum(linear_losses * (linear_losses > 0))
        assert np.abs(loss_value - np_loss_values) < 1e-3

    tf.reset_default_graph()

if __name__ == '__main__':
    pytest.main([__file__])
