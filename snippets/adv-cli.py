#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import math
import sys
import os

import numpy as np
import tensorflow as tf

from inferbeddings.io import read_triples
from inferbeddings.knowledgebase import Fact, KnowledgeBaseParser

from inferbeddings.parse import parse_clause

from inferbeddings.models import base as models
from inferbeddings.models import similarities

from inferbeddings.models.training import pairwise_losses, constraints, corrupt, index
from inferbeddings.models.training.util import make_batches

from inferbeddings import evaluation

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def train(session, train_sequences,
          nb_entities, nb_predicates, nb_batches, seed, similarity_name,
          entity_embedding_size, predicate_embedding_size,
          model_name, objective_name, margin, learning_rate, nb_epochs,
          parser, clauses):

    index_gen = index.GlorotIndexGenerator()
    neg_idxs = np.arange(nb_entities)

    subject_corruptor = corrupt.SimpleCorruptor(index_generator=index_gen, candidate_indices=neg_idxs, corrupt_objects=False)
    object_corruptor = corrupt.SimpleCorruptor(index_generator=index_gen, candidate_indices=neg_idxs, corrupt_objects=True)

    # Saving training examples in two Numpy matrices, Xr (nb_samples, 1) containing predicate ids,
    # and Xe (nb_samples, 2), containing subject and object ids.
    Xr = np.array([[rel_idx] for (rel_idx, _) in train_sequences])
    Xe = np.array([ent_idxs for (_, ent_idxs) in train_sequences])

    nb_samples = Xr.shape[0]

    # Number of samples per batch.
    batch_size = math.ceil(nb_samples / nb_batches)

    # Input for the subject and object ids.
    entity_inputs = tf.placeholder(tf.int32, shape=[None, 2])

    # Input for the predicate id - at the moment it is a length-1 walk, i.e. a predicate id only,
    # but it can correspond to a sequence of predicates (a walk in the knowledge graph).
    walk_inputs = tf.placeholder(tf.int32, shape=[None, None])

    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    tf.set_random_seed(seed)

    # Instantiate the model
    similarity_function = similarities.get_function(similarity_name)

    entity_embedding_layer = tf.get_variable('entities', shape=[nb_entities + 1, entity_embedding_size],
                                             initializer=tf.contrib.layers.xavier_initializer())
    predicate_embedding_layer = tf.get_variable('predicates', shape=[nb_predicates + 1, predicate_embedding_size],
                                                initializer=tf.contrib.layers.xavier_initializer())

    entity_embeddings = tf.nn.embedding_lookup(entity_embedding_layer, entity_inputs)
    predicate_embeddings = tf.nn.embedding_lookup(predicate_embedding_layer, walk_inputs)

    model_class = models.get_function(model_name)

    model = model_class(entity_embeddings, predicate_embeddings, similarity_function,
                        entity_embedding_size=entity_embedding_size, predicate_embedding_size=predicate_embedding_size)

    # Scoring function used for scoring arbitrary triples.
    score = model()

    nb_clauses = len(clauses)

    # For each clause, instantiate head and one body violators
    arg1_violator_layer = tf.get_variable('arg1_violator', shape=[nb_clauses, entity_embedding_size], initializer=tf.contrib.layers.xavier_initializer())
    arg2_violator_layer = tf.get_variable('arg2_violator', shape=[nb_clauses, entity_embedding_size], initializer=tf.contrib.layers.xavier_initializer())

    arg1_violator = tf.expand_dims(arg1_violator_layer, 1)
    arg2_violator = tf.expand_dims(arg2_violator_layer, 1)

    clause_head_predicate_idxs = []
    clause_body_predicate_idxs = []
    for clause in clauses:
        head_predicate = clause.head.predicate.name
        body_predicate = clause.body[0].predicate.name

        clause_head_predicate_idxs += [parser.predicate_to_index[head_predicate]]
        clause_body_predicate_idxs += [parser.predicate_to_index[body_predicate]]

    clause_head_predicate_idxs = np.array(clause_head_predicate_idxs, dtype=np.int32)
    clause_body_predicate_idxs = np.array(clause_body_predicate_idxs, dtype=np.int32)

    clause_head_predicate_inputs = tf.Variable(clause_head_predicate_idxs, name='clause_head_predicate_inputs')
    clause_body_predicate_inputs = tf.Variable(clause_body_predicate_idxs, name='clause_body_predicate_inputs')

    clause_head_predicate_embeddings = tf.nn.embedding_lookup(predicate_embedding_layer, clause_head_predicate_inputs)
    clause_body_predicate_embeddings = tf.nn.embedding_lookup(predicate_embedding_layer, clause_body_predicate_inputs)

    clause_head_predicates = tf.expand_dims(clause_head_predicate_embeddings, 1)
    clause_body_predicates = tf.expand_dims(clause_body_predicate_embeddings, 1)

    arg1_arg2_violator = tf.concat(1, [arg1_violator, arg2_violator])
    arg2_arg1_violator = tf.concat(1, [arg2_violator, arg1_violator])

    clause_head_scoring_model = model_class(arg1_arg2_violator, clause_head_predicates, similarity_function,
                                            entity_embedding_size=entity_embedding_size,
                                            predicate_embedding_size=predicate_embedding_size)
    clause_body_scoring_model = model_class(arg2_arg1_violator, clause_body_predicates, similarity_function,
                                            entity_embedding_size=entity_embedding_size,
                                            predicate_embedding_size=predicate_embedding_size)

    clause_head_scores = clause_head_scoring_model()
    clause_body_scores = clause_body_scoring_model()

    rules_violation_loss = tf.reduce_sum(tf.maximum(0.0, clause_body_scores - clause_head_scores))

    rules_violation_finding_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    rules_violation_training_step = rules_violation_finding_optimizer.minimize(- rules_violation_loss, var_list=[arg1_violator_layer, arg2_violator_layer])

    # Transform the pairwise loss function in an unary loss function,
    # where each positive example is followed by a negative example.
    def pairwise_to_unary_modifier(_loss_function):
        def unary_function(scores, *_args, **_kwargs):
            positive_scores, negative_scores = tf.split(1, 2, tf.reshape(scores, [-1, 2]))
            return _loss_function(positive_scores, negative_scores, *_args, **_kwargs)
        return unary_function

    # Loss function to minimize by means of Stochastic Gradient Descent.
    objective_function = pairwise_to_unary_modifier(pairwise_losses.get_function(objective_name))
    loss_function = objective_function(score, margin=margin) + 0 * rules_violation_loss

    # Optimization algorithm being used.
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)

    training_step = optimizer.minimize(loss_function, var_list=[entity_embedding_layer, predicate_embedding_layer])

    # We enforce all entity embeddings to have an unitary norm.
    constraining_step = constraints.renorm_update(entity_embedding_layer, norm=1.0)

    constraining_arg1_violators_step = constraints.renorm_update(arg1_violator_layer, norm=1.0)
    constraining_arg2_violators_step = constraints.renorm_update(arg2_violator_layer, norm=1.0)

    init_op = tf.initialize_all_variables()

    session.run(init_op)

    for epoch in range(1, nb_epochs + 1):
        order = random_state.permutation(nb_samples)
        Xr_shuf, Xe_shuf = Xr[order, :], Xe[order, :]

        Xr_sc, Xe_sc = subject_corruptor(Xr_shuf, Xe_shuf)
        Xr_oc, Xe_oc = object_corruptor(Xr_shuf, Xe_shuf)

        batches = make_batches(nb_samples, batch_size)
        loss_values = []

        for batch_start, batch_end in batches:
            curr_batch_size = batch_end - batch_start

            Xr_batch = np.zeros((curr_batch_size * 4, Xr.shape[1]), dtype=Xr.dtype)
            Xe_batch = np.zeros((curr_batch_size * 4, Xe.shape[1]), dtype=Xe.dtype)

            Xr_batch[0::4, :] = Xr_batch[2::4, :] = Xr[batch_start:batch_end, :]
            Xe_batch[0::4, :] = Xe_batch[2::4, :] = Xe[batch_start:batch_end, :]

            Xr_batch[1::4, :], Xe_batch[1::4, :] = Xr_sc[batch_start:batch_end, :], Xe_sc[batch_start:batch_end, :]
            Xr_batch[3::4, :], Xe_batch[3::4, :] = Xr_oc[batch_start:batch_end, :], Xe_oc[batch_start:batch_end, :]

            loss_args = {walk_inputs: Xr_batch, entity_inputs: Xe_batch}

            _, loss_value = session.run([training_step, loss_function], feed_dict=loss_args)
            session.run([constraining_step])

            loss_values += [loss_value / Xr_batch.shape[0]]

        logger.info('Epoch: {}\tLoss: {} ± {}'.format(epoch, round(np.mean(loss_values), 4), round(np.std(loss_values), 4)))

        logger.info('Finding violators ..')

        for finding_epoch in range(1, 10 + 1):
            _, violation_loss_value = session.run([rules_violation_training_step, rules_violation_loss])

            logging.info('Epoch: {}\tViolation loss: {}'.format(epoch, round(violation_loss_value, 4)))

            session.run([constraining_arg1_violators_step])
            session.run([constraining_arg2_violators_step])

    def scoring_function(args):
        return session.run(score, feed_dict={walk_inputs: args[0], entity_inputs: args[1]})

    return scoring_function


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Knowledge Graph Walker', formatter_class=formatter)

    argparser.add_argument('--train', '-t', required=True, action='store', type=str, default=None)
    argparser.add_argument('--valid', '-v', action='store', type=str, default=None)
    argparser.add_argument('--test', '-T', action='store', type=str, default=None)

    argparser.add_argument('--lr', '-l', action='store', type=float, default=0.1)

    argparser.add_argument('--nb-batches', '-b', action='store', type=int, default=10)
    argparser.add_argument('--nb-epochs', '-e', action='store', type=int, default=100)

    argparser.add_argument('--model', '-m', action='store', type=str, default='DistMult', help='Model')
    argparser.add_argument('--similarity', '-s', action='store', type=str, default='dot', help='Similarity function')
    argparser.add_argument('--objective', '-o', action='store', type=str, default='hinge_loss', help='Loss function')
    argparser.add_argument('--margin', '-M', action='store', type=float, default=1.0, help='Margin')

    argparser.add_argument('--embedding-size', '--entity-embedding-size', '-k', action='store', type=int, default=10, help='Entity embedding size')
    argparser.add_argument('--predicate-embedding-size', '-p', action='store', type=int, default=None, help='Predicate embedding size')

    argparser.add_argument('--auc', '-a', action='store_true', help='Measure the predictive accuracy using AUC-PR and AUC-ROC')
    argparser.add_argument('--seed', '-S', action='store', type=int, default=0, help='Seed for the PRNG')

    argparser.add_argument('--clauses', '-c', action='store', type=str, default=None,
                           help='File containing background knowledge expressed as Horn clauses')

    args = argparser.parse_args(argv)

    train_path = args.train
    valid_path = args.valid
    test_path = args.test

    learning_rate = args.lr

    nb_batches = args.nb_batches
    nb_epochs = args.nb_epochs

    model_name = args.model
    similarity_name = args.similarity
    objective_name = args.objective
    margin = args.margin

    entity_embedding_size = args.embedding_size
    predicate_embedding_size = args.predicate_embedding_size

    if predicate_embedding_size is None:
        predicate_embedding_size = entity_embedding_size

    is_auc = args.auc
    seed = args.seed

    clauses_path = args.clauses

    assert train_path is not None

    pos_train_triples, _ = read_triples(train_path)

    pos_valid_triples, neg_valid_triples = None, None
    if valid_path:
        pos_valid_triples, neg_valid_triples = read_triples(valid_path)

    pos_test_triples, neg_test_triples = None, None
    if test_path:
        pos_test_triples, neg_test_triples = read_triples(test_path)

    def fact(s, p, o):
        return Fact(predicate_name=p, argument_names=[s, o])

    train_facts = [fact(s, p, o) for s, p, o in pos_train_triples]

    valid_facts = [fact(s, p, o) for s, p, o in pos_valid_triples] if pos_valid_triples is not None else []
    valid_facts_neg = [fact(s, p, o) for s, p, o in neg_valid_triples] if neg_valid_triples is not None else []

    test_facts = [fact(s, p, o) for s, p, o in pos_test_triples] if pos_test_triples is not None else []
    test_facts_neg = [fact(s, p, o) for s, p, o in neg_test_triples] if neg_test_triples is not None else []

    logger.info('#Training Triples: {}\t#Validation Triples: {}\t#Test Triples: {}'.format(len(train_facts), len(valid_facts), len(test_facts)))

    parser = KnowledgeBaseParser(train_facts + valid_facts + test_facts)

    nb_entities = len(parser.entity_vocabulary)
    nb_predicates = len(parser.predicate_vocabulary)

    logger.info('#Entities: {}\t#Predicates: {}'.format(nb_entities, nb_predicates))

    train_sequences = parser.facts_to_sequences(train_facts)

    valid_sequences = parser.facts_to_sequences(valid_facts)
    valid_sequences_neg = parser.facts_to_sequences(valid_facts_neg)

    test_sequences = parser.facts_to_sequences(test_facts)
    test_sequences_neg = parser.facts_to_sequences(test_facts_neg)

    # Parse the clauses
    clauses = None
    if clauses_path is not None:
        with open(clauses_path, 'r') as f:
            clauses_str = [line.strip() for line in f.readlines()]
        clauses = [parse_clause(clause_str) for clause_str in clauses_str]

    # Do not take up all the GPU memory, all the time.
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as session:
        scoring_function = train(session, train_sequences, nb_entities, nb_predicates, nb_batches, seed, similarity_name,
                                 entity_embedding_size, predicate_embedding_size, model_name, objective_name, margin, learning_rate, nb_epochs,
                                 parser, clauses)

        train_triples = [(s, p, o) for (p, [s, o]) in train_sequences]

        valid_triples = [(s, p, o) for (p, [s, o]) in valid_sequences]
        valid_triples_neg = [(s, p, o) for (p, [s, o]) in valid_sequences_neg]

        test_triples = [(s, p, o) for (p, [s, o]) in test_sequences]
        test_triples_neg = [(s, p, o) for (p, [s, o]) in test_sequences_neg]

        true_triples = train_triples + valid_triples + test_triples

        if valid_triples:
            if is_auc:
                evaluation.evaluate_auc(scoring_function, valid_triples, valid_triples_neg, nb_entities, nb_predicates, tag='valid')
            else:
                evaluation.evaluate_ranks(scoring_function, valid_triples, nb_entities, true_triples=true_triples, tag='valid')

        if test_triples:
            if is_auc:
                evaluation.evaluate_auc(scoring_function, test_triples, test_triples_neg, nb_entities, nb_predicates, tag='test')
            else:
                evaluation.evaluate_ranks(scoring_function, test_triples, nb_entities, true_triples=true_triples, tag='test')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
