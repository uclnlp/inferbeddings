#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    batch_size = 5
    n = 3
    walk_len = 2

    rs = np.random.RandomState(0)

    p = rs.rand(batch_size, walk_len, n)
    vp = tf.Variable(p)

    init_op = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init_op)

        _batch_size, _walk_len, _embedding_len = tf.shape(vp)[0],  tf.shape(vp)[1], tf.shape(vp)[2]
        transposed_embedding_matrix = tf.transpose(vp, perm=[1, 0, 2])

        initializer = tf.zeros((_batch_size, _embedding_len), dtype=vp.dtype)
        walk_embedding = tf.scan(tf.add, transposed_embedding_matrix, initializer=initializer)[-1]
        out = session.run(walk_embedding)

        print(out)

        print(np.sum(p, axis=1))
