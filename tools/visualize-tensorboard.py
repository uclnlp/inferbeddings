#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    embeddings = tf.get_variable('W', shape=[10, 100], initializer=tf.contrib.layers.xavier_initializer())
    init_op = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init_op)

        saver = tf.train.Saver()
        saver.save(session, "model.ckpt", 0)

        summary_writer = tf.summary.FileWriter('.')

        projector_config = projector.ProjectorConfig()

        embedding = projector_config.embeddings.add()
        embedding.tensor_name = embeddings.name

        projector.visualize_embeddings(summary_writer, projector_config)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
