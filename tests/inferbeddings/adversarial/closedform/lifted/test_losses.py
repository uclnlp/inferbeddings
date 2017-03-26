# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from inferbeddings.models import base as models
from inferbeddings.models import similarities
from inferbeddings.knowledgebase import Fact, KnowledgeBaseParser
from inferbeddings.parse import parse_clause
from inferbeddings.models.training import constraints

from inferbeddings.adversarial.ground import GroundLoss
from inferbeddings.adversarial.closedform import ClosedFormLifted

import logging

import pytest

logger = logging.getLogger(__name__)

triples = [
    ('a', 'p', 'b'),
    ('c', 'p', 'd'),
    ('a', 'q', 'b')
]
facts = [Fact(predicate_name=p, argument_names=[s, o]) for s, p, o in triples]
parser = KnowledgeBaseParser(facts)

nb_entities = len(parser.entity_to_index)
nb_predicates = len(parser.predicate_to_index)

# Clauses
clause_str = 'q(X, Y) :- p(Y, X)'
clauses = [parse_clause(clause_str)]

# Instantiating the model parameters
model_class = models.get_function('ComplEx')
similarity_function = similarities.get_function('dot')

unit_cube = False


@pytest.mark.closedform
def test_losses():
    for seed in range(1):
        tf.reset_default_graph()

        np.random.seed(seed)
        tf.set_random_seed(seed)

        entity_embedding_size = np.random.randint(low=1, high=5) * 2
        predicate_embedding_size = entity_embedding_size

        # Instantiating entity and predicate embedding layers
        entity_embedding_layer = tf.get_variable('entities',
                                                 shape=[nb_entities + 1, entity_embedding_size],
                                                 initializer=tf.contrib.layers.xavier_initializer())

        predicate_embedding_layer = tf.get_variable('predicates',
                                                    shape=[nb_predicates + 1, predicate_embedding_size],
                                                    initializer=tf.contrib.layers.xavier_initializer())

        entity_projection = constraints.unit_sphere(entity_embedding_layer, norm=1.0)
        if unit_cube:
            entity_projection = constraints.unit_cube(entity_embedding_layer)

        entity_inputs = tf.placeholder(tf.int32, shape=[None, 2])
        walk_inputs = tf.placeholder(tf.int32, shape=[None, None])

        entity_embeddings = tf.nn.embedding_lookup(entity_embedding_layer, entity_inputs)
        predicate_embeddings = tf.nn.embedding_lookup(predicate_embedding_layer, walk_inputs)

        model_parameters = dict(entity_embeddings=entity_embeddings,
                                predicate_embeddings=predicate_embeddings,
                                similarity_function=similarity_function)

        model = model_class(**model_parameters)
        score = model()

        closed_form_lifted = ClosedFormLifted(parser=parser,
                                              predicate_embedding_layer=predicate_embedding_layer,
                                              model_class=model_class,
                                              model_parameters=model_parameters,
                                              is_unit_cube=unit_cube)
        opt_adversarial_loss = closed_form_lifted(clauses[0])

        v_optimizer = tf.train.AdagradOptimizer(learning_rate=1e-1)
        v_training_step = v_optimizer.minimize(opt_adversarial_loss, var_list=[predicate_embedding_layer])

        init_op = tf.global_variables_initializer()

        with tf.Session() as session:

            def scoring_function(args):
                return session.run(score, feed_dict={walk_inputs: args[0], entity_inputs: args[1]})

            ground_loss = GroundLoss(clauses=clauses, parser=parser, scoring_function=scoring_function)

            entity_indices = sorted({idx for idx in parser.entity_to_index.values()})
            feed_dicts = GroundLoss.sample_mappings(GroundLoss.get_variable_names(clauses[0]),
                                                    entities=entity_indices,
                                                    sample_size=100)

            session.run(init_op)

            continuous_error_0 = ground_loss.continuous_errors(clauses[0], feed_dicts=feed_dicts)

            for finding_epoch in range(1, 100 + 1):
                _ = session.run([v_training_step])
                session.run([entity_projection])

            continuous_error_final = ground_loss.continuous_errors(clauses[0], feed_dicts=feed_dicts)

            assert continuous_error_final <= continuous_error_0

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__])
