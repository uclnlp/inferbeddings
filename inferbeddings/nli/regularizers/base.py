# -*- coding: utf-8 -*-

import tensorflow as tf


def contradiction_symmetry_l1(model_class, model_kwargs,
                              pooling_function=tf.reduce_sum,
                              contradiction_idx=1):
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

    inv_model = model_class(reuse=True, **model_kwargs)
    inv_logits = inv_model()
    inv_contradiction_prob = tf.nn.softmax(inv_logits)[:, contradiction_idx]

    diff = contradiction_prob - inv_contradiction_prob
    return pooling_function(abs(diff))


def contradiction_symmetry_l2(model_class, model_kwargs,
                              pooling_function=tf.reduce_sum,
                              contradiction_idx=1):
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

    inv_model = model_class(reuse=True, **model_kwargs)
    inv_logits = inv_model()
    inv_contradiction_prob = tf.nn.softmax(inv_logits)[:, contradiction_idx]

    diff = contradiction_prob - inv_contradiction_prob
    return pooling_function(diff ** 2)
