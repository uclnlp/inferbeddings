# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from inferbeddings.models import base as models
from inferbeddings.models import similarities
from inferbeddings.knowledgebase import Fact, KnowledgeBaseParser
from inferbeddings.parse import parse_clause
from inferbeddings.models.training import constraints

from inferbeddings.adversarial import Adversarial

import pytest

triples = [
    ('a', 'p', 'b'),
    ('c', 'p', 'd'),
    ('a', 'q', 'b')
]
facts = [Fact(predicate_name=p, argument_names=[s, o]) for s, p, o in triples]
parser = KnowledgeBaseParser(facts)

nb_entities = len(parser.entity_to_index)
nb_predicates = len(parser.predicate_to_index)

entity_embedding_size = 10
predicate_embedding_size = 10

# Clauses
clause_str = 'q(X, Y) :- p(X, Y)'
clauses = [parse_clause(clause_str)]

# Instantiating the model parameters
model_class = models.get_function('DistMult')
similarity_function = similarities.get_function('dot')

model_parameters = dict(similarity_function=similarity_function)


@pytest.mark.closedform
def test_distmult_unit_cube():
    for seed in range(32):
        tf.reset_default_graph()

        np.random.seed(seed)
        tf.set_random_seed(seed)

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

        adv_projection_steps = [constraints.unit_cube(adv_emb_layer) for adv_emb_layer in adversarial.parameters]

        v_errors, v_loss = adversarial.errors, adversarial.loss

        v_optimizer = tf.train.AdagradOptimizer(learning_rate=1e-1)
        v_training_step = v_optimizer.minimize(- v_loss, var_list=adversarial.parameters)

        init_op = tf.global_variables_initializer()

        p_idx, q_idx = parser.predicate_to_index['p'], parser.predicate_to_index['q']

        p_emb = tf.nn.embedding_lookup(predicate_embedding_layer, p_idx)
        q_emb = tf.nn.embedding_lookup(predicate_embedding_layer, q_idx)

        with tf.Session() as session:
            session.run(init_op)
            predicate_embedding_layer_value = session.run([predicate_embedding_layer])[0]

            # Analytically computing the best adversarial embeddings
            p_emb_val, q_emb_val = session.run([p_emb, q_emb])
            opt_emb = (p_emb_val >= q_emb_val).astype(np.float32)

            assert len(adversarial.parameters) == 2
            session.run([adversarial.parameters[0][0, :].assign(opt_emb)])
            session.run([adversarial.parameters[1][0, :].assign(opt_emb)])

            for projection_step in adv_projection_steps:
                session.run([projection_step])

            v_opt_errors_val, v_opt_loss_val = session.run([v_errors, v_loss])

            session.run(init_op)
            session.run([predicate_embedding_layer.assign(predicate_embedding_layer_value)])

            for finding_epoch in range(1, 100 + 1):
                _ = session.run([v_training_step])

                for projection_step in adv_projection_steps:
                    session.run([projection_step])

                v_errors_val, v_loss_val = session.run([v_errors, v_loss])

                print('{} <= {}'.format(v_loss_val, v_opt_loss_val))

                assert v_opt_errors_val >= v_errors_val
                assert v_opt_loss_val >= v_loss_val

if __name__ == '__main__':
    pytest.main([__file__])
