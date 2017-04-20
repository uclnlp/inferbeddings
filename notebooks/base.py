# -*- coding: utf-8 -*-

import math
import numpy as np
import tensorflow as tf

from inferbeddings.knowledgebase import Fact, KnowledgeBaseParser

from inferbeddings.models import base as models
from inferbeddings.models import similarities

from inferbeddings.models.training import losses, constraints, corrupt, index
from inferbeddings.models.training.util import make_batches

from inferbeddings.adversarial import Adversarial
import logging

logger = logging.getLogger(__name__)


class Inferbeddings:
    model_name = 'DistMult'
    similarity_name = 'dot'
    entity_embedding_size, predicate_embedding_size = 10, 10

    unit_cube = False
    nb_batches = 10

    def __init__(self, session, triples, clauses, random_state=None):
        self.triples, self.clauses = triples, clauses
        self.random_state = random_state or np.random.RandomState(seed=0)

        logger.info('Parsing the facts in the Knowledge Base ..')
        self.facts = [Fact(predicate_name=p, argument_names=[s, o]) for s, p, o in self.triples]
        self.parser = KnowledgeBaseParser(self.facts)

        self.nb_entities = len(self.parser.entity_vocabulary)
        self.nb_predicates = len(self.parser.predicate_vocabulary)

        self.__init_discriminator(session)
        self.__init_adversary(session, clauses)

    def __init_discriminator(self, session):
        logger.info('Initialising the Discriminator computational graph ..')
        self.entity_inputs = tf.placeholder(tf.int32, shape=[None, 2])
        self.walk_inputs = tf.placeholder(tf.int32, shape=[None, None])

        with tf.variable_scope('discriminator', reuse=False):
            self.entity_embedding_layer = tf.get_variable('entity_embeddings',
                                                          shape=[self.nb_entities + 1, self.entity_embedding_size],
                                                          initializer=tf.contrib.layers.xavier_initializer())
            self.predicate_embedding_layer = tf.get_variable('predicate_embeddings',
                                                             shape=[self.nb_predicates + 1, self.predicate_embedding_size],
                                                             initializer=tf.contrib.layers.xavier_initializer())

        discriminator_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
        initialize_discriminator = tf.variables_initializer(discriminator_variables)
        session.run([initialize_discriminator])

        self.entity_embeddings = tf.nn.embedding_lookup(self.entity_embedding_layer, self.entity_inputs)
        self.predicate_embeddings = tf.nn.embedding_lookup(self.predicate_embedding_layer, self.walk_inputs)

        self.model_class = models.get_function(self.model_name)
        self.similarity_function = similarities.get_function(self.similarity_name)

        self.model_parameters = dict(entity_embeddings=self.entity_embeddings,
                                     predicate_embeddings=self.predicate_embeddings,
                                     similarity_function=self.similarity_function)
        self.model = self.model_class(**self.model_parameters)

        # Scoring function used for scoring arbitrary triples.
        self.score = self.model()

        logger.info('Instantiating the fact loss function computational graph ..')
        self.nb_versions = 3
        hinge_loss = losses.get_function('hinge')

        # array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, ...], dtype=int32)
        target = (tf.range(0, limit=tf.shape(self.score)[0]) % self.nb_versions) < 1
        self.fact_loss = hinge_loss(self.score, tf.cast(target, self.score.dtype), margin=1)

    def init_adversary(self, session, clauses):
        self.__init_adversary(session, clauses, reuse=True)

    def __init_adversary(self, session, clauses, reuse=False):
        self.loss_function = self.fact_loss

        logger.info('Initialising the Adversary computational graph ..')

        self.adversaries = []
        for clause_idx, (clause, clause_weight) in enumerate(clauses):
            with tf.variable_scope('adversary/{}'.format(clause_idx), reuse=reuse):
                adversarial = Adversarial(clauses=[clause], parser=self.parser,
                                          entity_embedding_layer=self.entity_embedding_layer,
                                          predicate_embedding_layer=self.predicate_embedding_layer,
                                          model_class=self.model_class, model_parameters=self.model_parameters,
                                          pooling='max', batch_size=1)
                self.adversaries += [adversarial]

                initialize_violators = tf.variables_initializer(var_list=adversarial.parameters)
                session.run([initialize_violators])

                violation_loss = adversarial.loss

            scope_name = 'adversary/{}/optimizer'.format(clause_idx)
            with tf.variable_scope(scope_name, reuse=reuse):
                violation_opt = tf.train.AdagradOptimizer(learning_rate=0.1)
                self.violation_training_step = violation_opt.minimize(- violation_loss, var_list=adversarial.parameters)

            adversarial_optimizer_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)
            initialize_violation_finder = tf.variables_initializer(adversarial_optimizer_variables)
            session.run([initialize_violation_finder])

            self.loss_function += clause_weight * violation_loss

        scope_name = 'discriminator/optimizer'
        with tf.variable_scope(scope_name, reuse=reuse):
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
            trainable_var_list = [self.entity_embedding_layer, self.predicate_embedding_layer] + self.model.get_params()
            self.discriminator_training_step = self.optimizer.minimize(self.loss_function, var_list=trainable_var_list)

        discriminator_optimizer_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)
        initialize_discriminator_optimizer_variables = tf.variables_initializer(discriminator_optimizer_variables)
        session.run([initialize_discriminator_optimizer_variables])

    def train_discriminator(self, session, nb_epochs=1):
        index_gen = index.GlorotIndexGenerator()
        neg_idxs = np.array(sorted(set(self.parser.entity_to_index.values())))

        subject_corruptor = corrupt.SimpleCorruptor(index_generator=index_gen, candidate_indices=neg_idxs,
                                                    corrupt_objects=False)
        object_corruptor = corrupt.SimpleCorruptor(index_generator=index_gen, candidate_indices=neg_idxs,
                                                   corrupt_objects=True)

        train_sequences = self.parser.facts_to_sequences(self.facts)

        Xr = np.array([[rel_idx] for (rel_idx, _) in train_sequences])
        Xe = np.array([ent_idxs for (_, ent_idxs) in train_sequences])

        nb_samples = Xr.shape[0]
        batch_size = math.ceil(nb_samples / self.nb_batches)
        logger.info("Samples: {}, no. batches: {} -> batch size: {}".format(nb_samples, self.nb_batches, batch_size))

        projection_steps = [constraints.unit_cube(self.entity_embedding_layer) if self.unit_cube
                            else constraints.unit_sphere(self.entity_embedding_layer, norm=1.0)]

        for epoch in range(1, nb_epochs + 1):
            order = self.random_state.permutation(nb_samples)
            Xr_shuf, Xe_shuf = Xr[order, :], Xe[order, :]

            Xr_sc, Xe_sc = subject_corruptor(Xr_shuf, Xe_shuf)
            Xr_oc, Xe_oc = object_corruptor(Xr_shuf, Xe_shuf)

            batches = make_batches(nb_samples, batch_size)

            loss_values = []
            total_fact_loss_value = 0

            for batch_start, batch_end in batches:
                curr_batch_size = batch_end - batch_start

                Xr_batch = np.zeros((curr_batch_size * self.nb_versions, Xr_shuf.shape[1]), dtype=Xr_shuf.dtype)
                Xe_batch = np.zeros((curr_batch_size * self.nb_versions, Xe_shuf.shape[1]), dtype=Xe_shuf.dtype)

                # Positive Example
                Xr_batch[0::self.nb_versions, :] = Xr_shuf[batch_start:batch_end, :]
                Xe_batch[0::self.nb_versions, :] = Xe_shuf[batch_start:batch_end, :]

                # Negative examples (corrupting subject)
                Xr_batch[1::self.nb_versions, :] = Xr_sc[batch_start:batch_end, :]
                Xe_batch[1::self.nb_versions, :] = Xe_sc[batch_start:batch_end, :]

                # Negative examples (corrupting object)
                Xr_batch[2::self.nb_versions, :] = Xr_oc[batch_start:batch_end, :]
                Xe_batch[2::self.nb_versions, :] = Xe_oc[batch_start:batch_end, :]

                # Safety check - each positive example is followed by two negative (corrupted) examples
                assert Xr_batch[0] == Xr_batch[1] == Xr_batch[2]
                assert Xe_batch[0, 0] == Xe_batch[2, 0] and Xe_batch[0, 1] == Xe_batch[1, 1]

                loss_args = {self.walk_inputs: Xr_batch, self.entity_inputs: Xe_batch}

                train_steps = [self.discriminator_training_step, self.loss_function, self.fact_loss]
                _, loss_value, fact_loss_value = session.run(train_steps, feed_dict=loss_args)

                loss_values += [loss_value / (Xr_batch.shape[0] / self.nb_versions)]
                total_fact_loss_value += fact_loss_value

                for projection_step in projection_steps:
                    session.run([projection_step])

    def train_adversary(self, session, nb_epochs=1):
        projs = [constraints.unit_cube(adv_embedding_layer) if self.unit_cube
                 else constraints.unit_sphere(adv_embedding_layer, norm=1.0)
                 for a in self.adversaries for adv_embedding_layer in a.parameters]

        # Initialize the violating embeddings using real embeddings
        def ground_init_op(violating_embeddings):
            _ent_indices = np.array(sorted(self.parser.index_to_entity.keys()))
            rnd_ent_indices = _ent_indices[self.random_state.randint(low=0, high=len(_ent_indices),
                                                                     size=self.adversaries[0].batch_size)]
            _ent_embeddings = tf.nn.embedding_lookup(self.entity_embedding_layer, rnd_ent_indices)
            return violating_embeddings.assign(_ent_embeddings)

        assignment_ops = [ground_init_op(violating_emb) for a in self.adversaries for violating_emb in a.parameters]
        session.run(assignment_ops)

        for projection_step in projs:
            session.run([projection_step])

        for epoch in range(1, nb_epochs + 1):
            session.run([self.violation_training_step])

            for projection_step in projs:
                session.run([projection_step])

    def get_embeddings(self, session):
        """
        returns dict mapping entity symbols to embeddings,
        and dict mapping predicate symbols to embeddings, in current session.
        """

        ent_embeddings = session.run(self.entity_embedding_layer)
        pred_embeddings = session.run(self.predicate_embedding_layer)

        print(ent_embeddings.shape, pred_embeddings.shape)

        ent_to_emb = {ent:list(ent_embeddings[i,:]) for i, ent in self.parser.index_to_entity.items()}
        pred_to_emb = {pred:list(pred_embeddings[i,:]) for i, pred in self.parser.index_to_predicate.items()}

        return ent_to_emb, pred_to_emb
