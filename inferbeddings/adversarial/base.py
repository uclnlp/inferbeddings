# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
from inferbeddings.models.training import pairwise_losses
import logging

logger = logging.getLogger(__name__)


class Adversarial:
    """
    Utility class for, given a set of clauses, computing the symbolic violation loss.
    """

    def __init__(self, clauses, parser, predicate_embedding_layer,
                 model_class, model_parameters, loss_function=None, loss_margin=0.0, batch_size=1):

        self.clauses, self.parser = clauses, parser
        self.predicate_embedding_layer = predicate_embedding_layer

        self.entity_embedding_size = model_parameters['entity_embedding_size']

        self.model_class, self.model_parameters = model_class, model_parameters
        self.loss_function = loss_function

        self.batch_size = batch_size

        if self.loss_function is None:
            # Default continuous violation loss: tf.nn.relu(margin - head_scores + body_scores)
            self.loss_function = lambda body_scores, head_scores: pairwise_losses.max_hinge_loss(head_scores,
                                                                                                 body_scores,
                                                                                                 margin=loss_margin)

        # Symbolic functions computing the number of ground errors and the continuous loss
        self.errors, self.loss = 0, .0

        # Trainable parameters of the adversarial model
        self.parameters = []

        # Weight terms of clauses, as mapping from clause to term
        self.weights = {}

        # Mapping from clause to

        for clause_idx, clause in enumerate(clauses):
            clause_errors, clause_loss, clause_parameters = self._parse_clause('clause_{}'.format(clause_idx), clause)

            self.errors += clause_errors
            self.loss += clause_loss
            self.parameters += clause_parameters

    def _parse_atom(self, atom, variable_name_to_layer):
        """
        Given an atom in the form p(X, Y), where X and Y are associated to two distinct [1, k] embedding layers,
        return the symbolic score of the atom.
        """
        predicate_idx = self.parser.predicate_to_index[atom.predicate.name]
        # [batch_size x 1 x embedding_size] tensor
        walk_embeddings = tf.nn.embedding_lookup(self.predicate_embedding_layer, [[predicate_idx]] * self.batch_size)
        arg1_name, arg2_name = atom.arguments[0].name, atom.arguments[1].name

        # [batch_size x embedding_size] variables
        arg1_layer, arg2_layer = variable_name_to_layer[arg1_name], variable_name_to_layer[arg2_name]
        # [batch_size x 2 x embedding_size] tensor
        arg1_arg2_embeddings = tf.concat(1, [tf.expand_dims(arg1_layer, 1), tf.expand_dims(arg2_layer, 1)])

        model_parameters = self.model_parameters
        model_parameters['entity_embeddings'] = arg1_arg2_embeddings
        model_parameters['predicate_embeddings'] = walk_embeddings

        scoring_model = self.model_class(reuse_variables=True, **model_parameters)
        atom_score = scoring_model()

        return atom_score if not atom.negated else -1.0 * atom_score

    def _parse_conjunction(self, atoms, variable_name_to_layer):
        """
        Given a conjunction of atoms in the form p(X0, X1), q(X2, X3), r(X4, X5), return its symbolic score.
        """
        conjunction_score = None
        for atom in atoms:
            atom_score = self._parse_atom(atom, variable_name_to_layer=variable_name_to_layer)
            conjunction_score = atom_score if conjunction_score is None else tf.minimum(conjunction_score, atom_score)
        return conjunction_score

    def create_violator_variable(self, clause_name, variable_name):
        return tf.get_variable('{}_{}_violator'.format(clause_name, variable_name),
                               shape=[self.batch_size, self.entity_embedding_size],
                               initializer=tf.contrib.layers.xavier_initializer())

    def create_violator_placeholder(self, clause_name, variable_name):
        return tf.placeholder(dtype=tf.float32,
                              shape=[self.batch_size, self.entity_embedding_size],
                              name='{}_{}_violator_ph'.format(clause_name, variable_name))

    def _parse_clause(self, name, clause):
        """
        Given a clause in the form p(X0, X1) :- q(X2, X3), r(X4, X5), return its symbolic score.
        """
        head, body = clause.head, clause.body

        # Enumerate all variables
        variable_names = {argument.name for argument in head.arguments}
        for body_atom in body:
            variable_names |= {argument.name for argument in body_atom.arguments}

        # Instantiate a new layer for each variable
        variable_name_to_layer = dict()
        for variable_name in sorted(variable_names):
            # [batch_size, embedding_size] variable
            variable_layer = tf.get_variable('{}_{}_violator'.format(name, variable_name),
                                             shape=[self.batch_size, self.entity_embedding_size],
                                             initializer=tf.contrib.layers.xavier_initializer())
            # variable_layer = violator_creator(name, variable_name)
            variable_name_to_layer[variable_name] = variable_layer

        head_score = tf.maximum(0.0, self._parse_atom(head, variable_name_to_layer=variable_name_to_layer))
        body_score = self._parse_conjunction(body, variable_name_to_layer=variable_name_to_layer)

        parameters = [variable_name_to_layer[variable_name] for variable_name in sorted(variable_names)]

        errors = tf.reduce_sum(tf.cast(body_score > head_score, tf.float32))
        loss = self.loss_function(body_score, head_score)

        # learnable clause weights
        if clause.weight != 1.0:
            negation_variable_name_to_layer = dict()
            for variable_name in sorted(variable_names):
                # [batch_size, embedding_size] variable
                variable_layer = tf.get_variable('{}_{}_violator_negation'.format(name, variable_name),
                                                 shape=[self.batch_size, self.entity_embedding_size],
                                                 initializer=tf.contrib.layers.xavier_initializer())
                negation_variable_name_to_layer[variable_name] = variable_layer
            negation_head_score = self._parse_atom(head, variable_name_to_layer=negation_variable_name_to_layer)
            negation_body_score = self._parse_conjunction(body, variable_name_to_layer=negation_variable_name_to_layer)
            parameters += [negation_variable_name_to_layer[variable_name] for variable_name in sorted(variable_names)]

            if clause.weight is None:
                weight_variable = tf.get_variable('{}_weight'.format(name),
                                                  shape=(),
                                                  initializer=tf.constant(0.0))

                # todo: the parameter must likely be registered somewhere to guarantee that weights are learned in the D-step.
                # todo: parameters.append(weight_variable)
                # todo: better to project when optimising
                prob = tf.sigmoid(weight_variable)
                self.weights[clause] = weight_variable
            else:
                prob = clause.weight

            # we define the negation of a clause as [score(head) <= score(body)] //strictly speaking it should be "<".
            loss = prob * loss + (1 - prob) * self.loss_function(negation_head_score, negation_body_score)

            # we leave the errors as is

        return errors, loss, parameters


class GenerativeAdversarial:
    """
    Utility class for, given a set of clauses, computing the symbolic violation loss.
    """

    def __init__(self, clauses, parser, predicate_embedding_layer,
                 model_class, model_parameters, loss_function=None, loss_margin=0.0, batch_size=1,
                 builder=None):

        self.clauses, self.parser = clauses, parser
        self.predicate_embedding_layer = predicate_embedding_layer

        self.entity_embedding_size = model_parameters['entity_embedding_size']

        self.model_class, self.model_parameters = model_class, model_parameters
        self.loss_function = loss_function

        self.batch_size = batch_size
        self.builder = builder if builder else point_mass_generator(self.batch_size, self.entity_embedding_size)

        if self.loss_function is None:
            # Default continuous violation loss: tf.nn.relu(margin - head_scores + body_scores)
            self.loss_function = lambda body_scores, head_scores: pairwise_losses.max_hinge_loss(head_scores,
                                                                                                 body_scores,
                                                                                                 margin=loss_margin)

        # Symbolic functions computing the number of ground errors and the continuous loss
        self.errors, self.loss = 0, .0

        # Trainable parameters of the adversarial model
        self.parameters = []

        # Weight terms of clauses, as mapping from clause to term
        self.weights = {}

        # Mapping from clause to

        for clause_idx, clause in enumerate(clauses):
            clause_errors, clause_loss, clause_parameters = self._parse_clause('clause_{}'.format(clause_idx), clause)

            self.errors += clause_errors
            self.loss += clause_loss
            self.parameters += clause_parameters

    def _parse_atom(self, atom, variable_name_to_layer):
        """
        Given an atom in the form p(X, Y), where X and Y are associated to two distinct [1, k] embedding layers,
        return the symbolic score of the atom.
        """
        predicate_idx = self.parser.predicate_to_index[atom.predicate.name]
        # [batch_size x 1 x embedding_size] tensor
        walk_embeddings = tf.nn.embedding_lookup(self.predicate_embedding_layer, [[predicate_idx]] * self.batch_size)
        arg1_name, arg2_name = atom.arguments[0].name, atom.arguments[1].name

        # [batch_size x embedding_size] variables
        arg1_layer, arg2_layer = variable_name_to_layer[arg1_name], variable_name_to_layer[arg2_name]
        # [batch_size x 2 x embedding_size] tensor
        arg1_arg2_embeddings = tf.concat(1, [tf.expand_dims(arg1_layer, 1), tf.expand_dims(arg2_layer, 1)])

        model_parameters = self.model_parameters
        model_parameters['entity_embeddings'] = arg1_arg2_embeddings
        model_parameters['predicate_embeddings'] = walk_embeddings

        scoring_model = self.model_class(reuse_variables=True, **model_parameters)
        atom_score = scoring_model()

        return atom_score if not atom.negated else -1.0 * atom_score

    def _parse_conjunction(self, atoms, variable_name_to_layer):
        """
        Given a conjunction of atoms in the form p(X0, X1), q(X2, X3), r(X4, X5), return its symbolic score.
        """
        conjunction_score = None
        for atom in atoms:
            atom_score = self._parse_atom(atom, variable_name_to_layer=variable_name_to_layer)
            conjunction_score = atom_score if conjunction_score is None else tf.minimum(conjunction_score, atom_score)
        return conjunction_score

    def create_violator_variable(self, clause_name, variable_name):
        return tf.get_variable('{}_{}_violator'.format(clause_name, variable_name),
                               shape=[self.batch_size, self.entity_embedding_size],
                               initializer=tf.contrib.layers.xavier_initializer())

    def _parse_clause(self, name, clause):
        """
        Given a clause in the form p(X0, X1) :- q(X2, X3), r(X4, X5), return its symbolic score.
        """
        head, body = clause.head, clause.body

        # Enumerate all variables
        variable_names = {argument.name for argument in head.arguments}
        for body_atom in body:
            variable_names |= {argument.name for argument in body_atom.arguments}

        # Instantiate a new layer for each variable
        variable_name_to_layer = self.builder(self, name, variable_names)

        head_score = tf.maximum(0.0, self._parse_atom(head, variable_name_to_layer=variable_name_to_layer))
        body_score = self._parse_conjunction(body, variable_name_to_layer=variable_name_to_layer)

        parameters = [variable_name_to_layer[variable_name] for variable_name in sorted(variable_names)]

        errors = tf.reduce_sum(tf.cast(body_score > head_score, tf.float32))
        loss = self.loss_function(body_score, head_score)

        # learnable clause weights
        if clause.weight != 1.0:
            negation_variable_name_to_layer = self.builder(self, name, variable_names)
            negation_head_score = self._parse_atom(head, variable_name_to_layer=negation_variable_name_to_layer)
            negation_body_score = self._parse_conjunction(body, variable_name_to_layer=negation_variable_name_to_layer)
            parameters += [negation_variable_name_to_layer[variable_name] for variable_name in sorted(variable_names)]

            if clause.weight is None:
                weight_variable = tf.get_variable('{}_weight'.format(name),
                                                  shape=(),
                                                  initializer=tf.constant(0.0))

                # todo: the parameter must likely be registered somewhere to guarantee that weights are learned in the D-step.
                # todo: parameters.append(weight_variable)
                # todo: better to project when optimising
                prob = tf.sigmoid(weight_variable)
                self.weights[clause] = weight_variable
            else:
                prob = clause.weight

            # we define the negation of a clause as [score(head) <= score(body)] //strictly speaking it should be "<".
            loss = prob * loss + (1 - prob) * self.loss_function(negation_head_score, negation_body_score)

            # we leave the errors as is

        return errors, loss, parameters


def point_mass_generator(batch_size, entity_embedding_size):
    def build_basic_logvar_to_tensor_mapping(name, variable_names, postfix=""):
        # Instantiate a new layer for each variable
        variable_name_to_layer = dict()
        for variable_name in sorted(variable_names):
            # [batch_size, embedding_size] variable
            variable_layer = tf.get_variable('{}_{}_violator{}'.format(name, variable_name, postfix),
                                             shape=[batch_size, entity_embedding_size],
                                             initializer=tf.contrib.layers.xavier_initializer())
            # variable_layer = violator_creator(name, variable_name)
            variable_name_to_layer[variable_name] = variable_layer
        return variable_name_to_layer

    return build_basic_logvar_to_tensor_mapping


def deep_generator(batch_size, entity_embedding_size, noise_dim=2):
    def build_generative_deep_network(name, variable_names, postfix=""):
        # Instantiate a new layer for each variable
        variable_name_to_layer = dict()
        intermediate_dim = noise_dim
        with tf.variable_scope("Clause {}{}".format(name, postfix)) as scope:
            noise_samples = tf.random_normal([batch_size, noise_dim])
            latent_layer = slim.fully_connected(noise_samples, intermediate_dim, scope=scope)
            for variable_name in variable_names:
                with tf.variable_scope("Variable {}".format(variable_name)) as log_var_scope:
                    output_layer = slim.fully_connected(latent_layer, entity_embedding_size, scope=log_var_scope)
                    variable_name_to_layer[variable_name] = output_layer

        return variable_name_to_layer

    return build_generative_deep_network
