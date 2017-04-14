# -*- coding: utf-8 -*-

import sys
import os

import time

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
                 model_name, similarity_name, unit_cube=True,
                 random_state=None):
        self.triples = triples
        self.entity_embedding_size, self.predicate_embedding_size = entity_embedding_size, predicate_embedding_size
        self.model_name, self.similarity_name, self.unit_cube = model_name, similarity_name, unit_cube
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

        model_class = models.get_function(model_name)
        model_parameters = dict(entity_embeddings=self.entity_embeddings,
                                predicate_embeddings=self.predicate_embeddings,
                                similarity_function=self.similarity_function)
        self.model = model_class(**model_parameters)

        # Scoring function used for scoring arbitrary triples.
        self.score = self.model()

    def train(self, session, nb_epochs=1,
              nb_discriminator_epochs=1,
              nb_adversary_epochs=1,
              nb_batches=10):
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

        for epoch in range(1, nb_epochs + 1):

