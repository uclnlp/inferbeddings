# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from inferbeddings.models.training import constraints

import logging
import pytest

logger = logging.getLogger(__name__)


@pytest.mark.light
def test_normalization():
    embedding_initializer = tf.contrib.layers.xavier_initializer()

    embedding_layer = tf.get_variable('embeddings', shape=[1024, 100], initializer=embedding_initializer)
    unit_sphere_embeddings = constraints.unit_sphere(embedding_layer, norm=1.0)

    init_op = tf.variables_initializer([embedding_layer])

    with tf.Session() as session:
        for _ in range(256):
            session.run(init_op)

            embeddings = session.run(embedding_layer)

            # Use TensorFlow for normalizing the embeddings
            session.run(unit_sphere_embeddings)
            normalized_v1 = session.run(embedding_layer)

            # Use NumPy for normalizing the embeddings
            normalized_v2 = embeddings / np.linalg.norm(embeddings, axis=1).reshape((-1, 1))

            np.testing.assert_allclose(normalized_v1, normalized_v2, rtol=1e-6)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    pytest.main([__file__])
