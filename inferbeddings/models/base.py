# -*- coding: utf-8 -*-

import abc

import tensorflow as tf
from inferbeddings.models import embeddings

import sys


class BaseModel(metaclass=abc.ABCMeta):
    def __init__(self, entity_embeddings=None, predicate_embeddings=None, similarity_function=None,
                 entity_embedding_size=None, predicate_embedding_size=None, reuse_variables=False,
                 *args, **kwargs):
        """
        Abstract class inherited by all models.

        :param entity_embeddings: (batch_size, 2, entity_embedding_size) Tensor.
        :param predicate_embeddings: (batch_size, walk_size, predicate_embedding_size) Tensor.
        :param similarity_function: similarity function.
        :param entity_embedding_size: size of the entity embeddings.
        :param predicate_embedding_size: size of the predicate embeddings.
        :param reuse_variables: States whether the variables within the model need to be reused.
        """
        self.entity_embeddings = entity_embeddings
        self.predicate_embeddings = predicate_embeddings
        self.similarity_function = similarity_function

        self.entity_embeddings_size = entity_embedding_size
        self.predicate_embeddings_size = predicate_embedding_size

        self.reuse_variables = reuse_variables

    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError

    def get_params(self):
        return []


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

        emb_size = self.entity_embeddings_size
        es_re, es_im = subject_embedding[:, :emb_size // 2], subject_embedding[:, emb_size // 2:]
        eo_re, eo_im = object_embedding[:, :emb_size // 2], object_embedding[:, emb_size // 2:]
        ew_re, ew_im = walk_embedding[:, :emb_size // 2], walk_embedding[:, emb_size // 2:]

        def dot3(arg1, rel, arg2):
            return self.similarity_function(arg1 * rel, arg2)

        score = dot3(es_re, ew_re, eo_re) + dot3(es_re, ew_im, eo_im) + dot3(es_im, ew_re, eo_im) - dot3(es_im, ew_im, eo_re)
        return score


class ERMLP(BaseModel):
    def __init__(self, hidden_size=None, f=tf.tanh, *args, **kwargs):
        """
        Implementation of the ER-MLP model described in [1, 2]

        [1] Dong, X. L. et al. - Knowledge Vault: A Web-Scale Approach to Probabilistic Knowledge Fusion - KDD 2014
        [2] Nickel, M. et al. - A Review of Relational Machine Learning for Knowledge Graphs - IEEE 2016
        """
        super().__init__(*args, **kwargs)
        self.f = f

        ent_emb_size = self.entity_embeddings_size
        pred_emb_size = self.predicate_embeddings_size
        ent_emb_size = ent_emb_size + ent_emb_size + pred_emb_size

        with tf.variable_scope("ERMLP", reuse=self.reuse_variables) as _:
            self.C = tf.get_variable('C', shape=[ent_emb_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer())
            self.w = tf.get_variable('w', shape=[hidden_size, 1], initializer=tf.contrib.layers.xavier_initializer())

    def __call__(self):
        """
        :return: (batch_size) Tensor containing the scores associated by the models to the walks.
        """
        subject_embedding, object_embedding = self.entity_embeddings[:, 0, :], self.entity_embeddings[:, 1, :]
        # This model is non-compositional in nature, so it might not be trivial to represent a walk embedding
        walk_embedding = self.predicate_embeddings[:, 0, :]

        e_ijk = tf.concat(1, [subject_embedding, object_embedding, walk_embedding])
        h_ijk = tf.matmul(e_ijk, self.C)
        f_ijk = tf.squeeze(tf.matmul(self.f(h_ijk), self.w), axis=1)

        return f_ijk

    def get_params(self):
        params = super().get_params() + [self.C, self.w]
        return params


# Aliases
TransE = TranslatingEmbeddings = TranslatingModel
DistMult = BilinearDiagonal = BilinearDiagonalModel
RESCAL = Bilinear = BilinearModel
ComplEx = ComplexE = ComplexModel
ER_MLP = ERMLP


def get_function(function_name):
    this_module = sys.modules[__name__]
    if not hasattr(this_module, function_name):
        raise ValueError('Unknown model: {}'.format(function_name))
    return getattr(this_module, function_name)
