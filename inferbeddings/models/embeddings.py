# -*- coding: utf-8 -*-

import tensorflow as tf


def additive_walk_embedding(predicate_embeddings):
    """
    Takes a walk, represented by a 3D Tensor with shape (batch_size, walk_length, embedding_length),
    and computes its embedding using a simple additive models.
    This method is roughly equivalent to:
    > walk_embedding = tf.reduce_prod(predicate_embeddings, axis=1)

    :param predicate_embeddings: 3D Tensor containing the embedding of the predicates in the walk.
    :return: 2D tensor of size (batch_size, embedding_length) containing the walk embeddings.
    """
    batch_size, embedding_len = tf.shape(predicate_embeddings)[0], tf.shape(predicate_embeddings)[2]

    # Transpose the (batch_size, walk_length, n) Tensor in a (walk_length, batch_size, n) Tensor
    transposed_embedding_matrix = tf.transpose(predicate_embeddings, perm=[1, 0, 2])

    # Define the initializer of the scan procedure - an all-zeros matrix
    initializer = tf.zeros((batch_size, embedding_len), dtype=predicate_embeddings.dtype)

    # The walk embeddings are given by the sum of the predicate embeddings
    # where zero is the neutral element wrt. the element-wise sum
    walk_embedding = tf.scan(lambda x, y: x + y, transposed_embedding_matrix, initializer=initializer)

    # Add the initializer as the first step in the scan sequence, in case the walk has zero-length
    return tf.concat(values=[tf.expand_dims(initializer, 0), walk_embedding], axis=0)[-1]


def bilinear_diagonal_walk_embedding(predicate_embeddings):
    """
    Takes a walk, represented by a 3D Tensor with shape (batch_size, walk_length, embedding_length),
    and computes its embedding using a simple bilinear diagonal models.
    This method is roughly equivalent to:
    > walk_embedding = tf.reduce_prod(predicate_embeddings, axis=1)

    :param predicate_embeddings: 3D Tensor containing the embedding of the predicates in the walk.
    :return: 2D tensor of size (batch_size, embedding_length) containing the walk embeddings.
    """
    batch_size, embedding_len = tf.shape(predicate_embeddings)[0], tf.shape(predicate_embeddings)[2]

    # Transpose the (batch_size, walk_length, n) Tensor in a (walk_length, batch_size, n) Tensor
    transposed_embedding_matrix = tf.transpose(predicate_embeddings, perm=[1, 0, 2])

    # Define the initializer of the scan procedure - an all-ones matrix
    # where one is the neutral element wrt. the element-wise product
    initializer = tf.ones((batch_size, embedding_len), dtype=predicate_embeddings.dtype)

    # The walk embeddings are given by the element-wise product of the predicate embeddings
    walk_embedding = tf.scan(lambda x, y: x * y, transposed_embedding_matrix, initializer=initializer)

    # Add the initializer as the first step in the scan sequence, in case the walk has zero-length
    return tf.concat(values=[tf.expand_dims(initializer, 0), walk_embedding], axis=0)[-1]


def bilinear_walk_embedding(predicate_embeddings, entity_embedding_size):
    """
    Takes a walk, represented by a 3D Tensor with shape (batch_size, walk_length, embedding_length),
    and computes its embedding using a simple bilinear models.

    :param predicate_embeddings: 3D Tensor containing the embedding of the predicates in the walk.
    :param entity_embedding_size: size of the entity embeddings.
    :return: 2D tensor of size (batch_size, entity_embedding_length, entity_embedding_length) containing the walk embeddings.
    """
    batch_size, embedding_len = tf.shape(predicate_embeddings)[0], tf.shape(predicate_embeddings)[2]
    walk_len = tf.shape(predicate_embeddings)[1]

    n = entity_embedding_size

    # Define a (entity_embedding_len, entity_embedding_len) identity matrix
    identity_matrix = tf.reshape(tf.diag(tf.ones(n, dtype=predicate_embeddings.dtype)), (1, n, n))

    # Replicate the identity matrix batch_size times
    initializer = tf.tile(identity_matrix, multiples=[batch_size, 1, 1])

    # Transform the (batch_size, walk_length, n ** 2) Tensor in a (walk_length, batch_size, n, n) Tensor
    reshapen_embedding_matrix = tf.reshape(predicate_embeddings, (batch_size, walk_len, n, n))
    transformed_embedding_matrix = tf.transpose(reshapen_embedding_matrix, perm=[1, 0, 2, 3])

    # The first step in the walk is the identity matrix (the neutral element wrt. the matrix product)'
    transformed_embedding_matrix = tf.concat(values=[tf.expand_dims(initializer, 0), transformed_embedding_matrix], axis=0)

    # The walk embeddings are given by the matrix multiplication of the predicate embeddings
    walk_embeddings = tf.scan(lambda x, y: tf.matmul(x, y), transformed_embedding_matrix, initializer=initializer)
    return walk_embeddings[-1]


def complex_walk_embedding(predicate_embeddings):
    """
    Takes a walk, represented by a 3D Tensor with shape (batch_size, walk_length, embedding_length),
    and returns its [:, 0, :] entry.

    TODO - find a more clever way of embedding walks using Complex Embeddings.

    :param predicate_embeddings: 3D Tensor containing the embedding of the predicates in the walk.
    :return: 2D tensor of size (batch_size, entity_embedding_length, entity_embedding_length) containing the walk embeddings.
    """
    batch_size, embedding_len = tf.shape(predicate_embeddings)[0], tf.shape(predicate_embeddings)[2]

    def p(x, y):
        return x * y

    def hermitian_product(x, y):
        x_re, x_im = tf.split(value=x, num_or_size_splits=2, axis=1)
        y_re, y_im = tf.split(value=y, num_or_size_splits=2, axis=1)
        return tf.concat(values=[p(x_re, y_re) + p(x_im, y_im), p(x_re, y_im) - p(x_im, y_re)], axis=1)

    # Transpose the (batch_size, walk_length, n) Tensor in a (walk_length, batch_size, n) Tensor
    transposed_embedding_matrix = tf.transpose(predicate_embeddings, perm=[1, 0, 2])

    neutral_element = tf.concat(values=[
        tf.ones((batch_size, embedding_len // 2), dtype=predicate_embeddings.dtype),
        tf.zeros((batch_size, embedding_len // 2), dtype=predicate_embeddings.dtype)
    ], axis=1)

    # Define the initializer of the scan procedure - the neutral element of the Hermitian product
    initializer = neutral_element

    # The walk embeddings are given by the element-wise product of the predicate embeddings
    walk_embedding = tf.scan(lambda x, y: hermitian_product(x, y), transposed_embedding_matrix, initializer=initializer)

    # Add the initializer as the first step in the scan sequence, in case the walk has zero-length
    return tf.concat(values=[tf.expand_dims(initializer, 0), walk_embedding], axis=0)[-1]
