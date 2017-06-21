# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from inferbeddings.models import base as models
from inferbeddings.models import similarities
from inferbeddings.knowledgebase import Fact, KnowledgeBaseParser
from inferbeddings.parse import parse_clause
from inferbeddings.models.training import constraints

from inferbeddings.adversarial import Adversarial
from inferbeddings.adversarial.closedform import ClosedForm

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

model_parameters = dict(similarity_function=similarity_function)


@pytest.mark.closedform
def test_complex_unit_sphere():
    for seed in range(32):
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

        # Adversary - used for computing the adversarial loss
        adversarial = Adversarial(clauses=clauses, parser=parser,
                                  entity_embedding_layer=entity_embedding_layer,
                                  predicate_embedding_layer=predicate_embedding_layer,
                                  model_class=model_class,
                                  model_parameters=model_parameters,
                                  batch_size=1)

        adv_projection_steps = [constraints.unit_sphere(adv_emb_layer) for adv_emb_layer in adversarial.parameters]

        adversarial_loss = adversarial.loss

        v_optimizer = tf.train.AdagradOptimizer(learning_rate=1e-1)
        v_training_step = v_optimizer.minimize(- adversarial_loss, var_list=adversarial.parameters)

        init_op = tf.global_variables_initializer()

        closed_form_lifted = ClosedForm(parser=parser,
                                        predicate_embedding_layer=predicate_embedding_layer,
                                        model_class=model_class, model_parameters=model_parameters,
                                        is_unit_cube=False)
        opt_adversarial_loss = closed_form_lifted(clauses[0])

        with tf.Session() as session:
            session.run(init_op)

            for finding_epoch in range(1, 100 + 1):
                _ = session.run([v_training_step])

                for projection_step in adv_projection_steps:
                    session.run([projection_step])

                violation_loss_val, opt_adversarial_loss_val = session.run([adversarial_loss, opt_adversarial_loss])

                if violation_loss_val + 1e-1 > opt_adversarial_loss_val:
                    print('{} <= {}'.format(violation_loss_val, opt_adversarial_loss_val))

                assert violation_loss_val <= (opt_adversarial_loss_val + 1e-4)

        tf.reset_default_graph()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__])
