# -*- coding: utf-8 -*-

import tensorflow as tf


def contradiction_symmetry_l1(model_class, model_kwargs,
                              pooling_function=tf.reduce_sum,
                              entailment_idx=0, neutral_idx=1, contradiction_idx=2,
                              debug=False):
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

    losses = abs(contradiction_prob - inv_contradiction_prob)
    loss = pooling_function(losses)

    return (loss, losses) if debug else loss


def contradiction_symmetry_l2(model_class, model_kwargs,
                              pooling_function=tf.reduce_sum,
                              entailment_idx=0, neutral_idx=1, contradiction_idx=2,
                              debug=False):
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

    losses = (contradiction_prob - inv_contradiction_prob) ** 2
    loss = pooling_function(losses)

    return (loss, losses) if debug else loss


def contradiction_kullback_leibler(model_class, model_kwargs,
                                   pooling_function=tf.reduce_sum,
                                   entailment_idx=0, neutral_idx=1, contradiction_idx=2,
                                   debug=False):
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
                                 entailment_idx=0, neutral_idx=1, contradiction_idx=2,
                                 debug=False):
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


def contradiction_acl(model_class, model_kwargs,
                      pooling_function=tf.reduce_sum,
                      entailment_idx=0, neutral_idx=1, contradiction_idx=2,
                      debug=False, is_bi=False):
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
    losses = tf.nn.relu(p_i - q_i) + (tf.nn.relu(q_i - p_i) if is_bi else 0)

    loss = pooling_function(losses)
    return (loss, losses) if debug else loss


def entailment_acl(model_class, model_kwargs,
                   pooling_function=tf.reduce_sum,
                   entailment_idx=0, neutral_idx=1, contradiction_idx=2,
                   debug=False, is_bi=False):
    model = model_class(reuse=True, **model_kwargs)
    logits = model()

    entailment_prob = tf.nn.softmax(logits)[:, entailment_idx]
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
    inv_entailment_prob = tf.nn.softmax(inv_logits)[:, entailment_idx]

    p_i, q_i = entailment_prob, inv_contradiction_prob
    losses = tf.nn.relu(p_i - (1.0 - q_i))

    if is_bi:
        p_i, q_i = contradiction_prob, inv_entailment_prob
        losses += tf.nn.relu(q_i - (1.0 - p_i))

    loss = pooling_function(losses)
    return (loss, losses) if debug else loss


def neutral_acl(model_class, model_kwargs,
                pooling_function=tf.reduce_sum,
                entailment_idx=0, neutral_idx=1, contradiction_idx=2,
                debug=False, is_bi=False):
    model = model_class(reuse=True, **model_kwargs)
    logits = model()

    neutral_prob = tf.nn.softmax(logits)[:, neutral_idx]
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
    inv_neutral_prob = tf.nn.softmax(inv_logits)[:, neutral_idx]

    p_i, q_i = neutral_prob, inv_contradiction_prob
    losses = tf.nn.relu(p_i - (1.0 - q_i))

    if is_bi:
        p_i, q_i = contradiction_prob, inv_neutral_prob
        losses += tf.nn.relu(q_i - (1.0 - p_i))

    loss = pooling_function(losses)
    return (loss, losses) if debug else loss


def entailment_reflexive_acl(model_class, model_kwargs,
                pooling_function=tf.reduce_sum,
                entailment_idx=0, neutral_idx=1, contradiction_idx=2,
                debug=False, is_bi=False):

    sequence1, sequence1_length = model_kwargs['sequence1'], model_kwargs['sequence1_length']
    sequence2, sequence2_length = model_kwargs['sequence2'], model_kwargs['sequence2_length']

    model_kwargs_a = model_kwargs.copy()
    model_kwargs_b = model_kwargs.copy()

    model_kwargs_a['sequence1'], model_kwargs_a['sequence1_length'] = sequence1, sequence1_length
    model_kwargs_a['sequence2'], model_kwargs_a['sequence2_length'] = sequence1, sequence1_length

    model_kwargs_b['sequence1'], model_kwargs_b['sequence1_length'] = sequence2, sequence2_length
    model_kwargs_b['sequence2'], model_kwargs_b['sequence2_length'] = sequence2, sequence2_length

    model_a = model_class(reuse=True, **model_kwargs_a)
    model_b = model_class(reuse=True, **model_kwargs_b)

    logits_a = model_a()
    logits_b = model_b()

    entailment_a_prob = tf.nn.softmax(logits_a)[:, entailment_idx]
    entailment_b_prob = tf.nn.softmax(logits_b)[:, entailment_idx]

    losses = tf.nn.relu(1.0 - entailment_a_prob) + tf.nn.relu(1.0 - entailment_b_prob)

    loss = pooling_function(losses)
    return (loss, losses) if debug else loss
