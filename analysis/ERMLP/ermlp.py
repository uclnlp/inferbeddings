#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import sys
import os

import argparse
import numpy as np
import tensorflow as tf

from tqdm import tqdm

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))

entity_embedding_size = 20
predicate_embedding_size = 20

seed = 0
margin = 5

nb_epochs = 1000
nb_batches = 10

np.random.seed(seed)
random_state = np.random.RandomState(seed)
tf.set_random_seed(seed)


def read_triples(path):
    triples = []
    with open(path, 'rt') as f:
        for line in f.readlines():
            s, p, o = line.split()
            triples += [(s.strip(), p.strip(), o.strip())]
    return triples


def renorm_update(var_matrix, norm=1.0, axis=1):
    row_norms = tf.sqrt(tf.reduce_sum(tf.square(var_matrix), axis=axis))
    scaled = var_matrix * tf.expand_dims(norm / row_norms, axis=axis)
    return tf.assign(var_matrix, scaled)


def pseudoboolean_linear_update(var_matrix):
    pseudoboolean_linear = tf.minimum(1., tf.maximum(var_matrix, 0.))
    return tf.assign(var_matrix, pseudoboolean_linear)


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size / float(batch_size)))
    res = [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]
    return res


class ERMLP:
    def __init__(self, subject_embeddings=None, predicate_embeddings=None, object_embeddings=None,
                 hidden_size=1, f=tf.tanh):
        self.subject_embeddings, self.object_embeddings = subject_embeddings, object_embeddings
        self.predicate_embeddings = predicate_embeddings
        self.f, self.hidden_size = f, hidden_size

        subject_emb_size = self.subject_embeddings.get_shape()[-1].value
        predicate_emb_size = self.predicate_embeddings.get_shape()[-1].value
        object_emb_size = self.object_embeddings.get_shape()[-1].value

        input_size = subject_emb_size + object_emb_size + predicate_emb_size

        self.C = tf.get_variable('C', shape=[input_size, self.hidden_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
        self.w = tf.get_variable('w', shape=[self.hidden_size, 1],
                                 initializer=tf.contrib.layers.xavier_initializer())

    def __call__(self):
        e_ijk = tf.concat(values=[self.subject_embeddings, self.object_embeddings, self.predicate_embeddings], axis=1)
        h_ijk = tf.matmul(e_ijk, self.C)
        f_ijk = tf.squeeze(tf.matmul(self.f(h_ijk), self.w), axis=1)

        return f_ijk


class IndexGenerator:
    def __init__(self):
        self.random_state = np.random.RandomState(0)

    def __call__(self, n_samples, candidate_indices):
        shuffled_indices = candidate_indices[self.random_state.permutation(len(candidate_indices))]
        rand_ints = shuffled_indices[np.arange(n_samples) % len(shuffled_indices)]
        return rand_ints


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('ER-MLP KBP Model', formatter_class=formatter)
    argparser.add_argument('dataset', action='store', type=str, choices=['wn18', 'fb15k', 'fb122'])
    args = argparser.parse_args(argv)

    dataset_name = args.dataset

    train_triples = read_triples('{}/{}.triples.train'.format(dataset_name, dataset_name))
    valid_triples = read_triples('{}/{}.triples.valid'.format(dataset_name, dataset_name))
    test_triples = read_triples('{}/{}.triples.test'.format(dataset_name, dataset_name))

    all_triples = train_triples + valid_triples + test_triples

    entity_set = set([s for (s, p, o) in all_triples] + [o for (s, p, o) in all_triples])
    predicate_set = set([p for (s, p, o) in all_triples])

    nb_entities, nb_predicates = len(entity_set), len(predicate_set)
    nb_examples = len(train_triples)

    entity_to_idx = {entity: idx for idx, entity in enumerate(sorted(entity_set))}
    predicate_to_idx = {predicate: idx for idx, predicate in enumerate(sorted(predicate_set))}

    entity_embedding_layer = tf.get_variable('entities', shape=[nb_entities, entity_embedding_size],
                                             initializer=tf.contrib.layers.xavier_initializer())

    predicate_embedding_layer = tf.get_variable('predicates', shape=[nb_predicates, predicate_embedding_size],
                                                initializer=tf.contrib.layers.xavier_initializer())

    subject_inputs = tf.placeholder(tf.int32, shape=[None])
    predicate_inputs = tf.placeholder(tf.int32, shape=[None])
    object_inputs = tf.placeholder(tf.int32, shape=[None])

    target_inputs = tf.placeholder(tf.float32, shape=[None])

    subject_embeddings = tf.nn.embedding_lookup(entity_embedding_layer, subject_inputs)
    predicate_embeddings = tf.nn.embedding_lookup(predicate_embedding_layer, predicate_inputs)
    object_embeddings = tf.nn.embedding_lookup(entity_embedding_layer, object_inputs)

    model = ERMLP(subject_embeddings=subject_embeddings,
                  predicate_embeddings=predicate_embeddings,
                  object_embeddings=object_embeddings)

    scores = model()

    hinge_losses = tf.nn.relu(margin - scores * (2 * target_inputs - 1))
    loss = tf.reduce_sum(hinge_losses)

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    training_step = optimizer.minimize(loss)

    projection_step = pseudoboolean_linear_update(entity_embedding_layer)

    batch_size = math.ceil(nb_examples / nb_batches)
    batches = make_batches(nb_examples, batch_size)

    nb_versions = 3

    Xs = np.array([entity_to_idx[s] for (s, p, o) in train_triples], dtype=np.int32)
    Xp = np.array([predicate_to_idx[p] for (s, p, o) in train_triples], dtype=np.int32)
    Xo = np.array([entity_to_idx[o] for (s, p, o) in train_triples], dtype=np.int32)

    index_gen = IndexGenerator()

    init_op = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init_op)

        for epoch in range(nb_epochs + 1):
            order = random_state.permutation(nb_examples)
            Xs_shuf, Xp_shuf, Xo_shuf = Xs[order], Xp[order], Xo[order]

            for batch_no, (batch_start, batch_end) in enumerate(batches):
                curr_batch_size = batch_end - batch_start

                Xs_batch = np.zeros(curr_batch_size * nb_versions, dtype=Xs_shuf.dtype)
                Xp_batch = np.zeros(curr_batch_size * nb_versions, dtype=Xp_shuf.dtype)
                Xo_batch = np.zeros(curr_batch_size * nb_versions, dtype=Xo_shuf.dtype)

                Xs_batch[0::nb_versions] = Xs_shuf[batch_start:batch_end]
                Xp_batch[0::nb_versions] = Xp_shuf[batch_start:batch_end]
                Xo_batch[0::nb_versions] = Xo_shuf[batch_start:batch_end]

                # Xs_batch[1::nb_versions] needs to be corrupted
                Xs_batch[1::nb_versions] = index_gen(curr_batch_size, np.arange(nb_entities))
                Xp_batch[1::nb_versions] = Xp_shuf[batch_start:batch_end]
                Xo_batch[1::nb_versions] = Xo_shuf[batch_start:batch_end]

                # Xo_batch[2::nb_versions] needs to be corrupted
                Xs_batch[2::nb_versions] = Xs_shuf[batch_start:batch_end]
                Xp_batch[2::nb_versions] = Xp_shuf[batch_start:batch_end]
                Xo_batch[2::nb_versions] = index_gen(curr_batch_size, np.arange(nb_entities))

                feed_dict = {
                    subject_inputs: Xs_batch, predicate_inputs: Xp_batch, object_inputs: Xo_batch,
                    target_inputs: np.array([1.0, 0.0, 0.0] * curr_batch_size)
                }

                _, loss_value = session.run([training_step, loss], feed_dict=feed_dict)
                logger.info('Epoch {}/{} Loss value: {}'.format(epoch, batch_no, loss_value))

                session.run(projection_step)

        logger.info('Evaluating ..')

        for eval_name, eval_triples in [('valid', valid_triples), ('test', test_triples)]:

            ranks_subj, ranks_obj = [], []
            filtered_ranks_subj, filtered_ranks_obj = [], []

            scores_subj_lst, scores_obj_lst = [], []
            filtered_scores_subj_lst, filtered_scores_obj_lst = [], []
            eval_triples_idx = []

            for s, p, o in tqdm(eval_triples):
                s_idx, p_idx, o_idx = entity_to_idx[s], predicate_to_idx[p], entity_to_idx[o]

                Xs = np.full(shape=(nb_entities,), fill_value=s_idx, dtype=np.int32)
                Xp = np.full(shape=(nb_entities,), fill_value=p_idx, dtype=np.int32)
                Xo = np.full(shape=(nb_entities,), fill_value=o_idx, dtype=np.int32)

                feed_dict_corrupt_subj = {subject_inputs: np.arange(nb_entities), predicate_inputs: Xp, object_inputs: Xo}
                feed_dict_corrupt_obj = {subject_inputs: Xs, predicate_inputs: Xp, object_inputs: np.arange(nb_entities)}

                # scores of (1, p, o), (2, p, o), .., (N, p, o)
                scores_subj = session.run(scores, feed_dict=feed_dict_corrupt_subj)

                # scores of (s, p, 1), (s, p, 2), .., (s, p, N)
                scores_obj = session.run(scores, feed_dict=feed_dict_corrupt_obj)

                ranks_subj += [1 + np.sum(scores_subj > scores_subj[s_idx])]
                ranks_obj += [1 + np.sum(scores_obj > scores_obj[o_idx])]

                filtered_scores_subj = scores_subj.copy()
                filtered_scores_obj = scores_obj.copy()

                rm_idx_s = [entity_to_idx[fs] for (fs, fp, fo) in all_triples if fs != s and fp == p and fo == o]
                rm_idx_o = [entity_to_idx[fo] for (fs, fp, fo) in all_triples if fs == s and fp == p and fo != o]

                filtered_scores_subj[rm_idx_s] = - np.inf
                filtered_scores_obj[rm_idx_o] = - np.inf

                filtered_ranks_subj += [1 + np.sum(filtered_scores_subj > filtered_scores_subj[s_idx])]
                filtered_ranks_obj += [1 + np.sum(filtered_scores_obj > filtered_scores_obj[o_idx])]

                scores_subj_lst += [scores_subj]
                scores_obj_lst += [scores_obj]
                filtered_scores_subj_lst += [filtered_scores_subj]
                filtered_scores_obj_lst += [filtered_scores_obj]
                eval_triples_idx += [(s_idx, p_idx, o_idx)]

            save_me = {
                'scores_subj_lst': scores_subj_lst,
                'scores_obj_lst': scores_obj_lst,
                'ranks_subj': ranks_subj,
                'ranks_obj': ranks_obj,
                'filtered_ranks_subj': filtered_ranks_subj,
                'filtered_ranks_obj': filtered_ranks_obj,
                'filtered_scores_subj_lst': filtered_scores_subj_lst,
                'filtered_scores_obj_lst': filtered_scores_obj_lst,
                'entity_embeddings': session.run(entity_embedding_layer),
                'predicate_embeddings': session.run(predicate_embedding_layer),
            }

            with open('save.p', 'wb') as f:
                import pickle
                pickle.dump(save_me, f)

            ranks = ranks_subj + ranks_obj
            filtered_ranks = filtered_ranks_subj + filtered_ranks_obj

            for setting_name, setting_ranks in [('Raw', ranks), ('Filtered', filtered_ranks)]:
                mean_rank = np.mean(setting_ranks)
                logger.info('[{}] {} Mean Rank: {}'.format(eval_name, setting_name, mean_rank))
                for k in [1, 3, 5, 10]:
                    hits_at_k = np.mean(np.asarray(setting_ranks) <= k) * 100
                    logger.info('[{}] {} Hits@{}: {}'.format(eval_name, setting_name, k, hits_at_k))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
