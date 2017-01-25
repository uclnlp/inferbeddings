# -*- coding: utf-8 -*-

import pytest

import tensorflow as tf

from inferbeddings.knowledgebase import Fact, KnowledgeBaseParser
from inferbeddings.parse import parse_clause
from inferbeddings.models import base as models
from inferbeddings.models import similarities

from inferbeddings.adversarial import Adversarial


def test_adversarial():
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

    model_class = models.get_function('DistMult')

    similarity_function = similarities.get_function('dot')
    model_parameters = dict(similarity_function=similarity_function,
                            entity_embedding_size=entity_embedding_size,
                            predicate_embedding_size=predicate_embedding_size)

    batch_size = 10

    adversarial = Adversarial(clauses=clauses,
                              parser=parser,
                              predicate_embedding_layer=predicate_embedding_layer,
                              model_class=model_class,
                              model_parameters=model_parameters,
                              batch_size=batch_size)



if __name__ == '__main__':
    pytest.main([__file__])
