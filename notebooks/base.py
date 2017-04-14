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
                 model_name, similarity_name, unit_cube,
                 parser, clauses):
        self.triples = triples
        self.entity_embedding_size, self.predicate_embedding_size = entity_embedding_size, predicate_embedding_size
        self.model_name, self.similarity_name, self.unit_cube = model_name, similarity_name, unit_cube
        self.parser, self.clauses = parser, clauses

        def fact(s, p, o):
            return Fact(predicate_name=p, argument_names=[s, o])
        self.facts = [fact(s, p, o) for s, p, o in self.triples]
        self.parser = KnowledgeBaseParser(facts)

        self.nb_entities = len(parser.entity_vocabulary)
        self.nb_predicates = len(parser.predicate_vocabulary)

        self.entity_inputs = tf.placeholder(tf.int32, shape=[None, 2])
        self.walk_inputs = tf.placeholder(tf.int32, shape=[None, None])

        self.entity_embedding_layer = tf.get_variable('entities', shape=[nb_entities + 1, entity_embedding_size],
                                                      initializer=tf.contrib.layers.xavier_initializer())
        self.predicate_embedding_layer = tf.get_variable('predicates', shape=[nb_predicates + 1, predicate_embedding_size],
                                                         initializer=tf.contrib.layers.xavier_initializer())

        model_class = models.get_function(model_name)
        model_parameters = dict(entity_embeddings=entity_embeddings,
                                predicate_embeddings=predicate_embeddings,
                                similarity_function=similarity_function)
        self.model = model_class(**model_parameters)


