#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

import sys
import logging


def main(argv):
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    sess = tf.InteractiveSession()

    # Create the models
    x = tf.placeholder(tf.float32, [None, 784])
    W1 = tf.get_variable('W1', shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1', shape=[1, 256], initializer=tf.contrib.layers.xavier_initializer())

    W2 = tf.get_variable('W2', shape=[256, 10], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable('b2', shape=[1, 10], initializer=tf.contrib.layers.xavier_initializer())

    y = tf.nn.softmax(tf.matmul(tf.sigmoid(tf.matmul(x, W1) + b1), W2) + b2)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(- tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.AdagradOptimizer(0.01).minimize(cross_entropy)

    # Train
    tf.initialize_all_variables().run()
    for i in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(10000)
        train_step.run({x: batch_xs, y_: batch_ys})

        # Test trained models
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
