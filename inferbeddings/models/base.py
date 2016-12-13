# -*- coding: utf-8 -*-

import abc

import tensorflow as tf
from inferbeddings.models import embeddings

import sys


class BaseModel(metaclass=abc.ABCMeta):
    def __init__(self, entity_embeddings, predicate_embeddings, similarity_function,
                 entity_embedding_size=None, predicate_embedding_size=None):
        """
        Abstract class inheritedby all models.

        :param entity_embeddings: (batch_size, 2, entity_embedding_size) Tensor.
        :param predicate_embeddings: (batch_size, walk_size, predicate_embedding_size) Tensor.
        :param similarity_function: similarity function.
        :param entity_embedding_size: size of the entity embeddings.
        :param predicate_embedding_size: size of the predicate embeddings.
        """
        self.entity_embeddings = entity_embeddings
        self.predicate_embeddings = predicate_embeddings
        self.similarity_function = similarity_function

        self.entity_embeddings_size = entity_embedding_size
        self.predicate_embeddings_size = predicate_embedding_size

    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError


class TranslatingModel(BaseModel):
    def __init__(self, *args, **kwargs):
        """
        Implementation of a compositional extension of the Translating Embeddings model [1].

        [1] Bordes, A. et al. - Translating Embeddings for Modeling Multi-relational Data - NIPS 2013
        """
        super().__init__(*args, **kwargs)

    def __call__(self):
        """
        :return: (batch_size) Tensor containing the scores associated by the models to the walks.
        """
        subject_embedding, object_embedding = self.entity_embeddings[:, 0, :], self.entity_embeddings[:, 1, :]

        #walk_embedding = tf.reduce_sum(predicate_embeddings, reduction_indices=1)
        walk_embedding = embeddings.additive_walk_embedding(self.predicate_embeddings)

        translated_subject_embedding = subject_embedding + walk_embedding
        return self.similarity_function(translated_subject_embedding, object_embedding)


class BilinearDiagonalModel(BaseModel):
    def __init__(self, *args, **kwargs):
        """
        Implementation of a compositional extension of the Bilinear-Diagonal model [1]

        [1] Yang, B. et al. - Embedding Entities and Relations for Learning and Inference in Knowledge Bases - ICLR 2015
        """
        super().__init__(*args, **kwargs)

    def __call__(self):
        """
        :return: (batch_size) Tensor containing the scores associated by the models to the walks.
        """
        subject_embedding, object_embedding = self.entity_embeddings[:, 0, :], self.entity_embeddings[:, 1, :]

        #walk_embedding = tf.reduce_prod(predicate_embeddings, reduction_indices=1)
        walk_embedding = embeddings.bilinear_diagonal_walk_embedding(self.predicate_embeddings)

        scaled_subject_embedding = subject_embedding * walk_embedding
        return self.similarity_function(scaled_subject_embedding, object_embedding)


class BilinearModel(BaseModel):
    def __init__(self, *args, **kwargs):
        """
        Implementation of a compositional extension of the Bilinear model [1]

        [1] Nickel, M. et al. - A Three-Way Model for Collective Learning on Multi-Relational Data - ICML 2011
        """
        super().__init__(*args, **kwargs)

    def __call__(self):
        """
        :return: (batch_size) Tensor containing the scores associated by the models to the walks.
        """
        subject_embedding, object_embedding = self.entity_embeddings[:, 0, :], self.entity_embeddings[:, 1, :]
        walk_embedding = embeddings.bilinear_walk_embedding(self.predicate_embeddings, self.entity_embeddings_size)

        es = tf.expand_dims(subject_embedding, 1)
        sW = tf.batch_matmul(es, walk_embedding)[:, 0, :]

        return self.similarity_function(sW, object_embedding)


class ComplexModel(BaseModel):
    def __init__(self, *args, **kwargs):
        """
        Implementation of a compositional extension of the ComplEx model [1]

        [1] Trouillon, T. et al. - Complex Embeddings for Simple Link Prediction - ICML 2016
        """
        super().__init__(*args, **kwargs)

    def __call__(self):
        """
        :return: (batch_size) Tensor containing the scores associated by the models to the walks.
        """
        subject_embedding, object_embedding = self.entity_embeddings[:, 0, :], self.entity_embeddings[:, 1, :]
        walk_embedding = embeddings.complex_walk_embedding(self.predicate_embeddings, self.entity_embeddings_size)

        n = self.entity_embeddings_size

        es_re, es_im = subject_embedding[:, :n // 2], subject_embedding[:, n // 2:]
        eo_re, eo_im = object_embedding[:, :n // 2], object_embedding[:, n // 2:]
        ew_re, ew_im = walk_embedding[:, :n // 2], walk_embedding[:, n // 2:]

        def dot3(arg1, rel, arg2):
            return self.similarity_function(arg1 * rel, arg2)

        score = dot3(es_re, ew_re, eo_re) + dot3(es_re, ew_im, eo_im) + dot3(es_im, ew_re, eo_im) - dot3(es_im, ew_im, eo_re)
        return score


# Aliases
TransE = TranslatingEmbeddings = TranslatingModel
DistMult = BilinearDiagonal = BilinearDiagonalModel
RESCAL = Bilinear = BilinearModel
ComplEx = ComplexE = ComplexModel


def get_function(function_name):
    this_module = sys.modules[__name__]
    if not hasattr(this_module, function_name):
        raise ValueError('Unknown model: {}'.format(function_name))
    return getattr(this_module, function_name)
