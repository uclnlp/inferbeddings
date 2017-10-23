# -*- coding: utf-8 -*-

import tensorflow as tf


def contradiction_symmetry_l1(model_class, model_kwargs,
                              pooling_function=tf.reduce_sum,
                              contradiction_idx=1, debug=False):
    model = model_class(reuse=True, **model_kwargs)
    logits = model()

    contradiction_prob = tf.nn.softmax(logits)[:, contradiction_idx]

    inv_sequence2, inv_sequence2_length = model_kwargs['sequence1'], model_kwargs['sequence1_length']
    inv_sequence1, inv_sequence1_length = model_kwargs['sequence2'], model_kwargs['sequence2_length']

    inv_model_kwargs = model_kwargs.copy()

    inv_model_kwargs['sequence1'] = inv_sequence1
    inv_model_kwargs['sequence1_length'] = inv_sequence1_length

    inv_model_kwargs['sequence2'] = inv_sequence2
    inv_model_kwargs['sequence2_length'] = inv_sequence2_length

    inv_model = model_class(reuse=True, **inv_model_kwargs)
    inv_logits = inv_model()

    inv_contradiction_prob = tf.nn.softmax(inv_logits)[:, contradiction_idx]

    losses = contradiction_prob - inv_contradiction_prob
    loss = pooling_function(abs(losses))

    return (loss, losses) if debug else loss


def contradiction_symmetry_l2(model_class, model_kwargs,
                              pooling_function=tf.reduce_sum,
                              contradiction_idx=1, debug=False):
    model = model_class(reuse=True, **model_kwargs)
    logits = model()
    contradiction_prob = tf.nn.softmax(logits)[:, contradiction_idx]

    inv_sequence2, inv_sequence2_length = model_kwargs['sequence1'], model_kwargs['sequence1_length']
    inv_sequence1, inv_sequence1_length = model_kwargs['sequence2'], model_kwargs['sequence2_length']

    inv_model_kwargs = model_kwargs.copy()

    inv_model_kwargs['sequence1'] = inv_sequence1
    inv_model_kwargs['sequence1_length'] = inv_sequence1_length

    inv_model_kwargs['sequence2'] = inv_sequence2
    inv_model_kwargs['sequence2_length'] = inv_sequence2_length

    inv_model = model_class(reuse=True, **inv_model_kwargs)
    inv_logits = inv_model()

    inv_contradiction_prob = tf.nn.softmax(inv_logits)[:, contradiction_idx]

    losses = contradiction_prob - inv_contradiction_prob
    loss = pooling_function(losses ** 2)

    return (loss, losses) if debug else loss


def contradiction_kullback_leibler(model_class, model_kwargs,
                                   pooling_function=tf.reduce_sum,
                                   contradiction_idx=1, debug=False):
    model = model_class(reuse=True, **model_kwargs)
    logits = model()
    contradiction_prob = tf.nn.softmax(logits)[:, contradiction_idx]

    inv_sequence2, inv_sequence2_length = model_kwargs['sequence1'], model_kwargs['sequence1_length']
    inv_sequence1, inv_sequence1_length = model_kwargs['sequence2'], model_kwargs['sequence2_length']

    inv_model_kwargs = model_kwargs.copy()

    inv_model_kwargs['sequence1'] = inv_sequence1
    inv_model_kwargs['sequence1_length'] = inv_sequence1_length

    inv_model_kwargs['sequence2'] = inv_sequence2
    inv_model_kwargs['sequence2_length'] = inv_sequence2_length

    inv_model = model_class(reuse=True, **inv_model_kwargs)
    inv_logits = inv_model()

    inv_contradiction_prob = tf.nn.softmax(inv_logits)[:, contradiction_idx]

    p_i, q_i = contradiction_prob, inv_contradiction_prob
    losses = (p_i ** 2) / q_i + (q_i ** 2) / p_i

    loss = pooling_function(losses)
    return (loss, losses) if debug else loss


def contradiction_jensen_shannon(model_class, model_kwargs,
                                 pooling_function=tf.reduce_sum,
                                 contradiction_idx=1, debug=False):
    model = model_class(reuse=True, **model_kwargs)
    logits = model()
    contradiction_prob = tf.nn.softmax(logits)[:, contradiction_idx]

    inv_sequence2, inv_sequence2_length = model_kwargs['sequence1'], model_kwargs['sequence1_length']
    inv_sequence1, inv_sequence1_length = model_kwargs['sequence2'], model_kwargs['sequence2_length']

    inv_model_kwargs = model_kwargs.copy()

    inv_model_kwargs['sequence1'] = inv_sequence1
    inv_model_kwargs['sequence1_length'] = inv_sequence1_length

    inv_model_kwargs['sequence2'] = inv_sequence2
    inv_model_kwargs['sequence2_length'] = inv_sequence2_length

    inv_model = model_class(reuse=True, **inv_model_kwargs)
    inv_logits = inv_model()

    inv_contradiction_prob = tf.nn.softmax(inv_logits)[:, contradiction_idx]

    p_i, q_i = contradiction_prob, inv_contradiction_prob
    losses = (p_i ** 2) / (p_i + q_i) + (q_i ** 2) / (p_i + q_i)

    loss = pooling_function(losses)
    return (loss, losses) if debug else loss
