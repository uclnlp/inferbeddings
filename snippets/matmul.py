#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    batch_size, n = 2, 3

    rs = np.random.RandomState(0)

    x = rs.rand(batch_size, n)
    w = rs.rand(batch_size, n * n)

    vx = tf.Variable(x)
    vw = tf.Variable(w)

    init_op = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(init_op)

        vW = tf.reshape(vw, (batch_size, n, n))
        vxW = tf.squeeze(tf.batch_matmul(tf.expand_dims(vx, 1), vW), [1])

        out = session.run(vxW)

        print(out[0])
        print(np.dot(x[0], np.reshape(w[0], (n, n))))

    num_matrices = 1

    rs = np.random.RandomState(0)

    x = rs.rand(batch_size, n)
    w = rs.rand(batch_size, num_matrices, n * n)

    vx = tf.Variable(x)
    vw = tf.Variable(w)

    init_op = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(init_op)

        vW = tf.reshape(vw, (batch_size, num_matrices, n, n))

        c = lambda i, _x: i < 0 #tf.shape(vW)[1]
        b = lambda i, _x: [i + 1, tf.squeeze(tf.batch_matmul(_x, vW[:, i, :, :]), [1])]

        r = tf.while_loop(c, b, [tf.constant(0), tf.expand_dims(vx, 1)])

        out = session.run(r)

        print(out)
