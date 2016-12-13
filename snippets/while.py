#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    i = tf.constant(0)

    c = lambda x: tf.less(x, 10)
    b = lambda x: tf.add(x, 1)

    r = tf.while_loop(c, b, [i])

    init_op = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(init_op)
        out = session.run(r)

    print(out)
