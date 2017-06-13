# -*- coding: utf-8 -*-

import tensorflow as tf


class AdversarialSets:
    """
    Utility class for generating Adversarial Sets for RTE.
    """
    def __init__(self, model_class, model_kwargs,
                 scope_name='adversary', embedding_size=300, batch_size=1024, sequence_length=10,
                 entailment_idx=0, contradiction_idx=1, neutral_idx=2):
        self.model_class = model_class
        self.model_kwargs = model_kwargs

        self.scope_name = scope_name
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        self.entailment_idx = entailment_idx
        self.contradiction_idx = contradiction_idx
        self.neutral_idx = neutral_idx

    def _get_sequence(self, name):
        with tf.variable_scope(self.scope_name):
            res = tf.get_variable(name=name,
                                  shape=[self.batch_size, self.sequence_length, self.embedding_size],
                                  initializer=tf.contrib.layers.xavier_initializer())
        return res

    def _logit(self, sequence1, sequence2, predicate_idx):
        model_kwargs = self.model_kwargs.copy()

        batch_size = sequence1.get_shape()[0].value
        sequence_length = tf.fill(dims=(batch_size,), value=self.sequence_length)

        model_kwargs.update({
            'sequence1': sequence1, 'sequence1_length': sequence_length,
            'sequence2': sequence2, 'sequence2_length': sequence_length
        })

        logits = self.model_class(**model_kwargs, reuse=True)()
        probability = logits[:, predicate_idx]
        return probability

    def rule1(self):
        """
        Adversarial loss term enforcing the rule:
            P(contradicts(S1, S2)) ~ P(contradicts(S2, S1))
        by computing:
            [P(contradicts(S1, S2)) - P(contradicts(S2, S1))]^2,
        where the sentence embeddings S1 and S2 can be learned adversarially.
    
        :return: (tf.Tensor, Set[tf.Variable]) pair containing the adversarial loss
            and the adversarially trainable variables.
        """
        # S1 - [batch_size, time_steps, embedding_size] sentence embedding.
        sequence1 = self._get_sequence(name='rule1_sequence1')
        # S2 - [batch_size, time_steps, embedding_size] sentence embedding.
        sequence2 = self._get_sequence(name='rule1_sequence2')

        # Probability that S1 contradicts S2
        p_s1_contradicts_s2 = self._logit(sequence1, sequence2, self.contradiction_idx)
        # Probability that S2 contradicts S1
        p_s2_contradicts_s1 = self._logit(sequence2, sequence1, self.contradiction_idx)

        return tf.nn.l2_loss(p_s1_contradicts_s2 - p_s2_contradicts_s1), {sequence1, sequence2}

    def rule2(self):
        """
        Adversarial loss term enforcing the rule:
            entails(S1, S2), entails(S2, S3) \implies entails(S1, S3)
        or, in other terms:
            min(P(entails(S1, S2)) + P(entails(S2, S3))) <= P(entails(S1, S3))
        by computing:
            ReLU[min(P(entails(S1, S2)) + P(entails(S2, S3))) - P(entails(S1, S3))]
        where the sentence embeddings S1, S2 and S3 can be learned adversarially.

        :return: (tf.Tensor, Set[tf.Variable]) pair containing the adversarial loss
            and the adversarially trainable variables.
        """
        # S1 - [batch_size, time_steps, embedding_size] sentence embedding.
        sequence1 = self._get_sequence(name='rule2_sequence1')
        # S2 - [batch_size, time_steps, embedding_size] sentence embedding.
        sequence2 = self._get_sequence(name='rule2_sequence2')
        # S3 - [batch_size, time_steps, embedding_size] sentence embedding.
        sequence3 = self._get_sequence(name='rule2_sequence3')

        # Probability that S1 entails S2
        p_s1_entails_s2 = self._logit(sequence1, sequence2, self.entailment_idx)
        # Probability that S2 entails S3
        p_s2_entails_s3 = self._logit(sequence2, sequence3, self.entailment_idx)
        # Probability that S1 entails S3
        p_s1_entails_s3 = self._logit(sequence1, sequence3, self.entailment_idx)

        body_score = tf.minimum(p_s1_entails_s2, p_s2_entails_s3)
        head_score = p_s1_entails_s3

        # The loss is > 0 if min(P1->P2, P2->P3) > P1->P3, 0 otherwise
        loss = tf.nn.relu(body_score - head_score)

        return loss, {sequence1, sequence2, sequence3}
