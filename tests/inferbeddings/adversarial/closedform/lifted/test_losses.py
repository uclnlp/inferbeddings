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

import itertools
import logging

import pytest

logger = logging.getLogger(__name__)

triples = [
    ('a', 'p', 'b'),
    ('a', 'q', 'b')
]
facts = [Fact(predicate_name=p, argument_names=[s, o]) for s, p, o in triples]
parser = KnowledgeBaseParser(facts)

a_idx = parser.entity_to_index['a']
b_idx = parser.entity_to_index['b']

nb_entities = len(parser.entity_to_index)
nb_predicates = len(parser.predicate_to_index)


def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

hyperparams = {
    'unit_cube': [True, False],
    'model_name': ['DistMult', 'ComplEx'],
    'clause': ['q(X, Y) :- p(X, Y)', 'q(X, Y) :- p(Y, X)']
}


@pytest.mark.closedform
def test_losses():

    hyperparam_configurations = list(cartesian_product(hyperparams))

    for hyperparam_configuration in hyperparam_configurations:
        # Clauses
        clause = parse_clause(hyperparam_configuration['clause'])

        # Instantiating the model parameters
        model_class = models.get_function(hyperparam_configuration['model_name'])
        similarity_function = similarities.get_function('dot')

        unit_cube = hyperparam_configuration['unit_cube']

        for seed in range(4):
            print('Seed {}, Evaluating {}'.format(seed, str(hyperparam_configuration)))

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
            opt_adversarial_loss = closed_form_lifted(clause)

            v_optimizer = tf.train.AdagradOptimizer(learning_rate=1e-2)
            v_training_step = v_optimizer.minimize(opt_adversarial_loss, var_list=[predicate_embedding_layer])

            init_op = tf.global_variables_initializer()

            with tf.Session() as session:
                session.run(init_op)

                session.run([entity_projection])

                def scoring_function(args):
                    return session.run(score, feed_dict={walk_inputs: args[0], entity_inputs: args[1]})

                ground_loss = GroundLoss(clauses=[clause], parser=parser, scoring_function=scoring_function)
                feed_dict = {'X': a_idx, 'Y': b_idx}
                continuous_loss_0 = ground_loss.continuous_error(clause, feed_dict=feed_dict)

                for epoch in range(1, 100 + 1):
                    _ = session.run([v_training_step])
                    print(ground_loss.continuous_error(clause, feed_dict=feed_dict))

                continuous_loss_final = ground_loss.continuous_error(clause, feed_dict=feed_dict)

                assert continuous_loss_0 <= .0 or continuous_loss_final <= continuous_loss_0

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__])
