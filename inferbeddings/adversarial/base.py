# -*- coding: utf-8 -*-

import tensorflow as tf
from inferbeddings.models.training import pairwise_losses

import logging

logger = logging.getLogger(__name__)


class Adversarial:
    def __init__(self, clauses, parser,
                 entity_embedding_layer, predicate_embedding_layer,
                 entity_embedding_size, predicate_embedding_size,
                 similarity_function, model_class, loss_function=None, margin=0.0):
        self.clauses = clauses
        self.parser = parser

        self.entity_embedding_layer = entity_embedding_layer
        self.predicate_embedding_layer = predicate_embedding_layer

        self.entity_embedding_size = entity_embedding_size
        self.predicate_embedding_size = predicate_embedding_size

        self.similarity_function = similarity_function
        self.model_class = model_class

        self.loss_function = loss_function
        self.loss_margin = margin
        if self.loss_function is None:
            # Default violation loss
            self.loss_function = lambda bs, hs: pairwise_losses.hinge_loss(hs, bs, margin=margin)

        self.errors = 0
        self.loss = 0
        self.parameters = []

        for clause_idx, clause in enumerate(clauses):
            clause_errors, clause_loss, clause_parameters = self._parse_clause('clause_{}'.format(clause_idx), clause)

            self.errors += clause_errors
            self.loss += clause_loss
            self.parameters += clause_parameters

    def _parse_atom(self, atom, variable_name_to_layer):
        predicate_ids = tf.Variable([self.parser.predicate_to_index[atom.predicate.name]])
        predicate_embedding = tf.nn.embedding_lookup(self.predicate_embedding_layer, predicate_ids)
        walk_embedding = tf.expand_dims(predicate_embedding, 1)

        arg1_name, arg2_name = atom.arguments[0].name, atom.arguments[1].name
        arg1_layer, arg2_layer = variable_name_to_layer[arg1_name], variable_name_to_layer[arg2_name]

        arg1_arg2_embeddings = tf.concat(1, [tf.expand_dims(arg1_layer, 1), tf.expand_dims(arg2_layer, 1)])

        scoring_model = self.model_class(arg1_arg2_embeddings, walk_embedding, self.similarity_function,
                                         entity_embedding_size=self.entity_embedding_size,
                                         predicate_embedding_size=self.predicate_embedding_size)
        atom_score = scoring_model()
        return atom_score

    def _parse_conjunction(self, atoms, variable_name_to_layer):
        conjunction_score = None
        for atom in atoms:
            atom_score = self._parse_atom(atom, variable_name_to_layer=variable_name_to_layer)
            conjunction_score = atom_score if conjunction_score is None else tf.minimum(conjunction_score, atom_score)
        return conjunction_score

    def _parse_clause(self, name, clause):
        head, body = clause.head, clause.body

        # Enumerate all variables
        variable_names = {argument.name for argument in head.arguments}
        for body_atom in body:
            variable_names |= {argument.name for argument in body_atom.arguments}

        # Instantiate a new layer for each variable
        variable_name_to_layer = dict()
        for variable_name in variable_names:
            variable_layer = tf.get_variable('{}_{}_violator'.format(name, variable_name),
                                             shape=[1, self.entity_embedding_size],
                                             initializer=tf.contrib.layers.xavier_initializer())
            variable_name_to_layer[variable_name] = variable_layer

        head_score = self._parse_atom(head, variable_name_to_layer=variable_name_to_layer)
        body_score = self._parse_conjunction(body, variable_name_to_layer=variable_name_to_layer)

        parameters = [layer for _, layer in variable_name_to_layer.items()]

        errors = tf.reduce_sum(tf.cast(body_score > head_score, tf.float32))
        loss = self.loss_function(body_score, head_score)

        return errors, loss, parameters
