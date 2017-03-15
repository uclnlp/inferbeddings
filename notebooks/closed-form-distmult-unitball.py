#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from inferbeddings.models import base as models
from inferbeddings.models import similarities
from inferbeddings.knowledgebase import Fact, KnowledgeBaseParser
from inferbeddings.parse import parse_clause
from inferbeddings.models.training import constraints

from inferbeddings.adversarial import Adversarial

import os
import sys
import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    # Knowledge Base triples
    triples = [
        ('a', 'p', 'b'),
        ('c', 'p', 'd'),
        ('a', 'q', 'b')
    ]

    nb_entities = 4
    nb_predicates = 2

    entity_embedding_size = 10
    predicate_embedding_size = 10
    seed = 2

    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    tf.set_random_seed(seed)

    # Instantiating entity and predicate embedding layers
    entity_embedding_layer = tf.get_variable('entities',
                                             shape=[nb_entities + 1, entity_embedding_size],
                                             initializer=tf.contrib.layers.xavier_initializer())

    predicate_embedding_layer = tf.get_variable('predicates',
                                                shape=[nb_predicates + 1, predicate_embedding_size],
                                                initializer=tf.contrib.layers.xavier_initializer())

    # Instantiating the model parameters
    model_class = models.get_function('DistMult')
    similarity_function = similarities.get_function('dot')

    model_parameters = dict(similarity_function=similarity_function)

    facts = [Fact(predicate_name=p, argument_names=[s, o]) for s, p, o in triples]
    parser = KnowledgeBaseParser(facts)

    # Clauses
    clause_str = 'q(X, Y) :- p(X, Y)'
    clauses = [parse_clause(clause_str)]

    # Adversary - used for computing the adversarial loss
    adversarial = Adversarial(clauses=clauses, parser=parser,
                              entity_embedding_layer=entity_embedding_layer,
                              predicate_embedding_layer=predicate_embedding_layer,
                              model_class=model_class,
                              model_parameters=model_parameters,
                              batch_size=1)

    adv_projection_steps = [constraints.unit_ball(adv_emb_layer, norm=1.0) for adv_emb_layer in adversarial.parameters]

    v_errors, v_loss = adversarial.errors, adversarial.loss

    v_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    v_training_step = v_optimizer.minimize(- v_loss, var_list=adversarial.parameters)

    init_op = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init_op)

        for finding_epoch in range(1, 100 + 1):
            _ = session.run([v_training_step])

            for projection_step in adv_projection_steps:
                session.run([projection_step])

            v_errors_val, v_loss_val = session.run([v_errors, v_loss])
            logging.info('{}\t{}'.format(v_errors_val, v_loss_val))

        p_idx = parser.predicate_to_index['p']
        q_idx = parser.predicate_to_index['q']

        p_emb = tf.nn.embedding_lookup(predicate_embedding_layer, p_idx)
        q_emb = tf.nn.embedding_lookup(predicate_embedding_layer, q_idx)

        # Analytically computing the maximum value for the adversar
        J = tf.reduce_max(p_emb - q_emb, reduction_indices=[0])
        J_val = session.run([J])
        logging.info('Maximum value for adversarial loss: {}'.format(J_val))

        # Analytically computing the best adversarial embeddings
        p_emb_val, q_emb_val = session.run([p_emb, q_emb])
        j = (p_emb_val - q_emb_val).argmax(axis=0)

        opt_emb = np.array([0] * entity_embedding_size)
        opt_emb[j] = 1

        session.run([adversarial.parameters[0][0, :].assign(opt_emb)])
        session.run([adversarial.parameters[1][0, :].assign(opt_emb)])

        for projection_step in adv_projection_steps:
            session.run([projection_step])

        v_errors_val, v_loss_val = session.run([v_errors, v_loss])
        logging.info('{}\t{}'.format(v_errors_val, v_loss_val))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
