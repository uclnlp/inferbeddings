# -*- coding: utf-8 -*-

import sys
import os

import math
import numpy as np
import tensorflow as tf

from inferbeddings.io import read_triples, save
from inferbeddings.knowledgebase import Fact, KnowledgeBaseParser

from inferbeddings.parse import parse_clause

from inferbeddings.models import base as models
from inferbeddings.models import similarities

from inferbeddings.models.training import losses, pairwise_losses, constraints, corrupt, index
from inferbeddings.models.training.util import make_batches

from inferbeddings.adversarial import Adversarial, GroundLoss

from inferbeddings import evaluation

import logging

logger = logging.getLogger(__name__)


class Inferbeddings:
    def __init__(self, triples,
                 entity_embedding_size, predicate_embedding_size,
                 model_name, similarity_name,
                 random_state=None):
        self.triples = triples
        self.entity_embedding_size, self.predicate_embedding_size = entity_embedding_size, predicate_embedding_size
        self.model_name, self.similarity_name = model_name, similarity_name
        self.random_state = random_state or np.random.RandomState(seed=0)

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
        self.training_step = self.optimizer.minimize(self.loss_function, var_list=trainable_var_list)

    def train(self, session,
              unit_cube=True, nb_epochs=1, nb_discriminator_epochs=1, nb_adversary_epochs=1, nb_batches=10):
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

        entity_projection = constraints.unit_sphere(self.entity_embedding_layer, norm=1.0)
        if unit_cube:
            entity_projection = constraints.unit_cube(self.entity_embedding_layer)

        projection_steps = [entity_projection]

        for epoch in range(1, nb_epochs + 1):

            for disc_epoch in range(1, nb_discriminator_epochs + 1):
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

                    _, loss_value, fact_loss_value = session.run([self.training_step, self.loss_function, self.fact_loss],
                                                                 feed_dict=loss_args)

                    loss_values += [loss_value / (Xr_batch.shape[0] / self.nb_versions)]
                    total_fact_loss_value += fact_loss_value

                    for projection_step in projection_steps:
                        session.run([projection_step])
