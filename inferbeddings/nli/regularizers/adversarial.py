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

    def _score(self, sequence1, sequence2, predicate_idx):
        model_kwargs = self.model_kwargs.copy()

        batch_size = sequence1.get_shape()[0].value
        sequence_length = tf.fill(dims=(batch_size,), value=self.sequence_length)

        model_kwargs.update({
            'sequence1': sequence1, 'sequence1_length': sequence_length,
            'sequence2': sequence2, 'sequence2_length': sequence_length
        })

        logits = self.model_class(reuse=True, **model_kwargs)()
        return logits[:, predicate_idx]

    def rule1_loss(self):
        """
        Adversarial loss term enforcing the rule:
            s(contradicts(S1, S2)) ~ s(contradicts(S2, S1))
        by computing:
            abs[s(contradicts(S1, S2)) - s(contradicts(S2, S1))],
        where the sentence embeddings S1 and S2 can be learned adversarially.
    
        :return: (tf.Tensor, Set[tf.Variable]) pair containing the adversarial loss
            and the adversarially trainable variables.
        """
        # S1 - [batch_size, time_steps, embedding_size] sentence embedding.
        sequence1 = self._get_sequence(name='rule1_sequence1')
        # S2 - [batch_size, time_steps, embedding_size] sentence embedding.
        sequence2 = self._get_sequence(name='rule1_sequence2')

        # Probability that S1 contradicts S2
        score_s1_contradicts_s2 = self._score(sequence1, sequence2, self.contradiction_idx)
        # Probability that S2 contradicts S1
        score_s2_contradicts_s1 = self._score(sequence2, sequence1, self.contradiction_idx)
        return tf.abs(score_s1_contradicts_s2 - score_s2_contradicts_s1), {sequence1, sequence2}

    def rule2_loss(self):
        """
        Adversarial loss term enforcing the rule:
            entails(S1, S2), entails(S2, S3) \implies entails(S1, S3)
        or, in other terms:
            min(s(entails(S1, S2)) + s(entails(S2, S3))) <= s(entails(S1, S3))
        by computing:
            ReLU[min(s(entails(S1, S2)) + s(entails(S2, S3))) - s(entails(S1, S3))]
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
        score_s1_entails_s2 = self._score(sequence1, sequence2, self.entailment_idx)
        # Probability that S2 entails S3
        score_s2_entails_s3 = self._score(sequence2, sequence3, self.entailment_idx)
        # Probability that S1 entails S3
        score_s1_entails_s3 = self._score(sequence1, sequence3, self.entailment_idx)

        body_score = tf.minimum(score_s1_entails_s2, score_s2_entails_s3)
        head_score = score_s1_entails_s3

        # The loss is > 0 if min(P1->P2, P2->P3) > P1->P3, 0 otherwise
        loss = tf.nn.relu(body_score - head_score)
        return loss, {sequence1, sequence2, sequence3}

    def rule3_loss(self):
        """
        Adversarial loss term enforcing the rule:
            s(entails(S1, S1)) > s(contradicts(S1, S1))
            s(entails(S1, S1)) > s(neutral(S1, S1))
        by computing:
            ReLU[s(contradicts(S1, S1)) - s(entails(S1, S1))]
            + ReLU[s(neutral(S1, S1)) - s(entails(S1, S1))],
        where the sentence embeddings S1 and S2 can be learned adversarially.

        :return: (tf.Tensor, Set[tf.Variable]) pair containing the adversarial loss
            and the adversarially trainable variables.
        """
        # S1 - [batch_size, time_steps, embedding_size] sentence embedding.
        sequence1 = self._get_sequence(name='rule3_sequence1')

        score_s1_entails_s1 = self._score(sequence1, sequence1, self.entailment_idx)
        score_s1_contradicts_s1 = self._score(sequence1, sequence1, self.contradiction_idx)
        score_s1_neutral_s1 = self._score(sequence1, sequence1, self.neutral_idx)

        loss = tf.nn.relu(score_s1_contradicts_s1 - score_s1_entails_s1) +\
            tf.nn.relu(score_s1_neutral_s1 - score_s1_entails_s1)
        return loss
