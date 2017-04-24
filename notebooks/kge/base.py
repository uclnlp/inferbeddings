# -*- coding: utf-8 -*-

import math
import numpy as np
import tensorflow as tf

from inferbeddings.knowledgebase import Fact, KnowledgeBaseParser

from inferbeddings.models import base as models
from inferbeddings.models import similarities

from inferbeddings.models.training import losses, constraints, corrupt, index
from inferbeddings.models.training.util import make_batches

import logging

logger = logging.getLogger(__name__)


class KGEmbeddings:
    @property
    def model_name(self):
        return self._model_name

    @property
    def similarity_name(self):
        return self._similarity_name

    @property
    def entity_embedding_size(self):
        return self._entity_embedding_size

    @property
    def predicate_embedding_size(self):
        return self._predicate_embedding_size

    @property
    def parameters(self):
        return [self.entity_embedding_layer, self.predicate_embedding_layer] + self.model.parameters

    def __init__(self, triples, optimizer,
                 model_name='DistMult', similarity_name='dot',
                 entity_embedding_size=10, predicate_embedding_size=10,
                 random_state=None):
        self.triples = triples

        self._model_name = model_name
        self._similarity_name = similarity_name
        self._entity_embedding_size = entity_embedding_size
        self._predicate_embedding_size = predicate_embedding_size

        self.random_state = random_state or np.random.RandomState(seed=0)

        logger.info('Parsing the facts in the Knowledge Base ..')
        self.facts = [Fact(predicate_name=p, argument_names=[s, o]) for s, p, o in self.triples]
        self.parser = KnowledgeBaseParser(self.facts)

        self.nb_entities, self.nb_predicates = len(self.parser.entity_vocabulary), len(self.parser.predicate_vocabulary)

        logger.info('Initialising the Discriminator computational graph ..')
        self.entity_inputs = tf.placeholder(tf.int32, shape=[None, 2])
        self.walk_inputs = tf.placeholder(tf.int32, shape=[None, None])

        self.entity_embedding_layer = tf.get_variable('entity_embeddings',
                                                      shape=[self.nb_entities + 1, self.entity_embedding_size],
                                                      initializer=tf.contrib.layers.xavier_initializer())

        self.predicate_embedding_layer = tf.get_variable('predicate_embeddings',
                                                         shape=[self.nb_predicates + 1, self.predicate_embedding_size],
                                                         initializer=tf.contrib.layers.xavier_initializer())

        self.entity_embeddings = tf.nn.embedding_lookup(self.entity_embedding_layer, self.entity_inputs)
        self.predicate_embeddings = tf.nn.embedding_lookup(self.predicate_embedding_layer, self.walk_inputs)

        self.model_class = models.get_function(self.model_name)
        self.similarity_function = similarities.get_function(self.similarity_name)

        model_parameters = {
            'entity_embeddings': self.entity_embeddings,
            'predicate_embeddings': self.predicate_embeddings,
            'similarity_function': self.similarity_function
        }
        self.model = self.model_class(**model_parameters)

        # Scoring function used for scoring arbitrary triples.
        self.score = self.model()

        logger.info('Instantiating the fact loss function computational graph ..')
        hinge_loss = losses.get_function('hinge')

        self.nb_versions = 3

        # array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, ...], dtype=int32)
        target = (tf.range(0, limit=tf.shape(self.score)[0]) % self.nb_versions) < 1
        self.fact_loss = hinge_loss(self.score, tf.cast(target, self.score.dtype), margin=1)

        self.training_step = optimizer.minimize(self.fact_loss, var_list=self.parameters)

    def train(self, session, unit_cube=True, nb_epochs=1, nb_batches=10):
        index_gen = index.GlorotIndexGenerator()
        neg_idxs = np.array(sorted(set(self.parser.entity_to_index.values())))

        subj_corruptor = corrupt.SimpleCorruptor(index_generator=index_gen, candidate_indices=neg_idxs, corrupt_objects=False)
        obj_corruptor = corrupt.SimpleCorruptor(index_generator=index_gen, candidate_indices=neg_idxs, corrupt_objects=True)

        train_sequences = self.parser.facts_to_sequences(self.facts)

        Xr = np.array([[rel_idx] for (rel_idx, _) in train_sequences])
        Xe = np.array([ent_idxs for (_, ent_idxs) in train_sequences])

        nb_samples = Xr.shape[0]
        batch_size = math.ceil(nb_samples / nb_batches)
        logger.info("Samples: {}, no. batches: {} -> batch size: {}".format(nb_samples, nb_batches, batch_size))

        projection_steps = [constraints.unit_cube(self.entity_embedding_layer) if unit_cube
                            else constraints.unit_sphere(self.entity_embedding_layer, norm=1.0)]

        for epoch in range(1, nb_epochs + 1):
            order = self.random_state.permutation(nb_samples)
            Xr_shuf, Xe_shuf = Xr[order, :], Xe[order, :]

            Xr_sc, Xe_sc = subj_corruptor(Xr_shuf, Xe_shuf)
            Xr_oc, Xe_oc = obj_corruptor(Xr_shuf, Xe_shuf)

            batches = make_batches(nb_samples, batch_size)

            loss_values = []
            total_loss_value = 0

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

                _, loss_value = session.run([self.training_step, self.fact_loss], feed_dict=loss_args)

                loss_values += [loss_value / (Xr_batch.shape[0] / self.nb_versions)]
                total_loss_value += loss_value

                for projection_step in projection_steps:
                    session.run([projection_step])

            def stats(values):
                return '{0:.4f} ± {1:.4f}'.format(round(np.mean(values), 4), round(np.std(values), 4))

            logger.info('Epoch: {0}\tLoss: {1}'.format(epoch, stats(loss_values)))

    def get_embeddings(self, session):
        """
        returns dict mapping entity symbols to embeddings,
        and dict mapping predicate symbols to embeddings, in current session.
        """
        ent_embeddings, pred_embeddings = session.run([self.entity_embedding_layer, self.predicate_embedding_layer])
        print(ent_embeddings.shape, pred_embeddings.shape)

        ent_to_emb = {ent: list(ent_embeddings[i, :]) for i, ent in self.parser.index_to_entity.items()}
        pred_to_emb = {pred: list(pred_embeddings[i, :]) for i, pred in self.parser.index_to_predicate.items()}

        return ent_to_emb, pred_to_emb

    def write_embeddings(self, session, pred_file=None, ent_file=None):
        ent_to_emb, pred_to_emb = self.get_embeddings(session)

        if pred_file:
            with open(pred_file, 'w') as fID:
                for pred in sorted(pred_to_emb.keys()):
                    fID.write('%s\t%s\n' % (pred, ','.join(['%.8f' % c for c in pred_to_emb[pred]])))
        if ent_file:
            with open(ent_file, 'w') as fID:
                for ent in sorted(ent_to_emb.keys()):
                    fID.write('%s\t%s\n' % (ent, ','.join(['%.8f' % c for c in ent_to_emb[ent]])))
