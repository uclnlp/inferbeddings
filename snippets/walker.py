#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import sys
import logging


def model(entity_inputs, walk_inputs):
    e_var = tf.get_variable('e_emb', shape=[1000, 10], initializer=tf.contrib.layers.xavier_initializer())
    p_var = tf.get_variable('p_emb', shape=[1000, 10], initializer=tf.contrib.layers.xavier_initializer())

    subj_obj_embeddings = tf.nn.embedding_lookup(e_var, entity_inputs)
    walk_embeddings = tf.nn.embedding_lookup(p_var, walk_inputs)

    subj_embedding = subj_obj_embeddings[:, 0, :]
    obj_embedding = subj_obj_embeddings[:, 1, :]

    walk_embedding = tf.reduce_sum(walk_embeddings, 1)
    score = - tf.reduce_sum(tf.abs(subj_embedding + walk_embedding - obj_embedding), 1)
    return score


def ranking_margin_objective(output, margin=10.0):
    y_pairs = tf.reshape(output, [-1, 2])
    pos_scores, neg_scores = tf.split(1, 2, y_pairs)
    hinge_losses = tf.nn.relu(margin - pos_scores + neg_scores)
    total_hinge_loss = tf.reduce_sum(hinge_losses)
    return output, total_hinge_loss


def main(argv):
    graph = tf.Graph()

    Xe = np.array([[0, 1], [0, 2], [0, 3], [0, 4]])
    Xr = np.array([[0], [0], [0], [0]])

    optimizer = tf.train.AdagradOptimizer(.1)

    with graph.as_default():
        tf.set_random_seed(0)

        entity_inputs = tf.placeholder(tf.int32, shape=[None, 2])
        walk_inputs = tf.placeholder(tf.int32, shape=[None, None])

        score = model(entity_inputs, walk_inputs)
        _, loss = ranking_margin_objective(score)
        train_step = optimizer.minimize(tf.reduce_sum(loss))

        init = tf.initialize_all_variables()

        with tf.Session(graph=graph) as sess:
            init.run()

            for i in range(16):
                batch_dict = {entity_inputs: Xe, walk_inputs: Xr}
                sess.run(train_step, batch_dict)
                print(sess.run(loss, batch_dict))

            batch_dict = {entity_inputs: Xe, walk_inputs: Xr}
            print(sess.run(score, batch_dict))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
