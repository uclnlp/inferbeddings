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
    def __init__(self, triples,
                 entity_embedding_size, predicate_embedding_size,
                 model_name, similarity_name,
                 clauses=None, adv_weight=1, random_state=None):
        self.triples = triples
        self.entity_embedding_size, self.predicate_embedding_size = entity_embedding_size, predicate_embedding_size
        self.model_name, self.similarity_name = model_name, similarity_name
        self.random_state = random_state or np.random.RandomState(seed=0)

        if clauses is None:
            clauses = []

        def fact(s, p, o):
            return Fact(predicate_name=p, argument_names=[s, o])
        self.facts = [fact(s, p, o) for s, p, o in self.triples]
        self.parser = KnowledgeBaseParser(self.facts)

        self.nb_entities = len(self.parser.entity_vocabulary)
        self.nb_predicates = len(self.parser.predicate_vocabulary)

        self.entity_inputs = tf.placeholder(tf.int32, shape=[None, 2])
        self.walk_inputs = tf.placeholder(tf.int32, shape=[None, None])

        self.entity_embedding_layer = tf.get_variable('entities',
                                                      shape=[self.nb_entities + 1, self.entity_embedding_size],
                                                      initializer=tf.contrib.layers.xavier_initializer())
        self.predicate_embedding_layer = tf.get_variable('predicates',
                                                         shape=[self.nb_predicates + 1, self.predicate_embedding_size],
                                                         initializer=tf.contrib.layers.xavier_initializer())

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
        self.nb_versions = 3

        hinge_loss = losses.get_function('hinge')

        # array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, ...], dtype=int32)
        target = (tf.range(0, limit=tf.shape(self.score)[0]) % self.nb_versions) < 1
        self.fact_loss = hinge_loss(self.score, tf.cast(target, self.score.dtype), margin=1)

        self.loss_function = self.fact_loss

        trainable_var_list = [self.entity_embedding_layer, self.predicate_embedding_layer] + self.model.get_params()
        self.optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        self.discriminator_training_step = self.optimizer.minimize(self.loss_function, var_list=trainable_var_list)

    def init_adversary(self, clauses, adv_weight=1, adv_lr=0.1, adv_batch_size=1):
        self.adversarial = Adversarial(clauses=clauses, parser=self.parser,
                                       entity_embedding_layer=self.entity_embedding_layer,
                                       predicate_embedding_layer=self.predicate_embedding_layer,
                                       model_class=self.model_class, model_parameters=self.model_parameters,
                                       pooling='max', batch_size=adv_batch_size)

        self.initialize_violators = tf.variables_initializer(var_list=self.adversarial.parameters, name='init_violators')
        self.violation_loss = self.adversarial.loss

        adv_opt_scope_name = 'adversarial/optimizer'
        with tf.variable_scope(adv_opt_scope_name):
            violation_opt = tf.train.AdagradOptimizer(learning_rate=adv_lr)
            self.violation_training_step = violation_opt.minimize(-violation_loss, var_list=self.adversarial.parameters)

        adversarial_optimizer_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=adv_opt_scope_name)
        self.adversarial_optimizer_variables_initializer = tf.variables_initializer(adversarial_optimizer_variables)

        self.loss_function = self.fact_loss + adv_weight * self.violation_loss

    def train_discriminator(self, session,
                            unit_cube=True, nb_epochs=1, nb_batches=10):
        index_gen = index.GlorotIndexGenerator()
        neg_idxs = np.array(sorted(set(self.parser.entity_to_index.values())))

        subject_corruptor = corrupt.SimpleCorruptor(index_generator=index_gen,
                                                    candidate_indices=neg_idxs,
                                                    corrupt_objects=False)
        object_corruptor = corrupt.SimpleCorruptor(index_generator=index_gen,
                                                   candidate_indices=neg_idxs,
                                                   corrupt_objects=True)

        train_sequences = self.parser.facts_to_sequences(self.facts)

        Xr = np.array([[rel_idx] for (rel_idx, _) in train_sequences])
        Xe = np.array([ent_idxs for (_, ent_idxs) in train_sequences])

        nb_samples = Xr.shape[0]

        batch_size = math.ceil(nb_samples / nb_batches)
        logger.info("Samples: %d, no. batches: %d -> batch size: %d" % (nb_samples, nb_batches, batch_size))

        if unit_cube:
            proj = constraints.unit_cube(self.entity_embedding_layer)
        else:
            proj = constraints.unit_sphere(self.entity_embedding_layer, norm=1.0)
        projection_steps = [proj]

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

    def train_adversary(self, session,
                        unit_cube=True, nb_epochs=1, adv_init_ground=True):
        if unit_cube:
            projs = [constraints.unit_cube(adv_embedding_layer) for adv_embedding_layer in self.adversarial.parameters]
        else:
            projs = [constraints.unit_sphere(adv_embedding_layer, norm=1.0) for adv_embedding_layer in self.adversarial.parameters]

        if adv_init_ground:
            # Initialize the violating embeddings using real embeddings
            def ground_init_op(violating_embeddings):
                # Select adv_batch_size random entity indices - first collect all entity indices
                _ent_indices = np.array(sorted(self.parser.index_to_entity.keys()))
                # Then select a subset of size adv_batch_size of such indices
                rnd_ent_indices = _ent_indices[self.random_state.randint(low=0, high=len(_ent_indices),
                                                                         size=self.adv_batch_size)]
                # Assign the embeddings of the entities at such indices to the violating embeddings
                _ent_embeddings = tf.nn.embedding_lookup(self.entity_embedding_layer, rnd_ent_indices)
                return violating_embeddings.assign(_ent_embeddings)

            assignment_ops = [ground_init_op(violating_emb) for violating_emb in self.adversarial.parameters]
            session.run(assignment_ops)

        for projection_step in projs:
            session.run([projection_step])

        for epoch in range(1, nb_epochs + 1):
            train_steps = [self.violation_training_step, self.violation_loss]
            _, violation_loss_value = session.run(train_steps)

            for projection_step in projs:
                session.run([projection_step])
