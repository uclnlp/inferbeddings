#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    batch_size, n = 1, 3
    num_matrices = 2

    rs = np.random.RandomState(0)

    x = rs.rand(batch_size, n)
    w = rs.rand(batch_size, num_matrices, n * n)

    vx = tf.Variable(x)
    vw = tf.Variable(w)

    init_op = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(init_op)

        vW = tf.reshape(vw, (batch_size, num_matrices, n, n))
        elems = tf.unpack(vW, axis=1)

        vsW = tf.foldl(tf.batch_matmul, elems)
        out = session.run(vsW)

        print(out)

    print(np.dot(np.reshape(w[:, 0], (n, n)), np.reshape(w[:, 1], (n, n))))
