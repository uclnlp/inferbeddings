#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


if __name__ == '__main__':

    rs = np.random.RandomState(0)
    x = rs.rand(5)

    vx = tf.Variable(x, name='x')

    init_op = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(init_op)

        vy = tf.tile(vx, multiples=[2, 2])
        out = session.run(vy)
        print(out)
