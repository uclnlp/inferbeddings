# -*- coding: utf-8 -*-

import tensorflow as tf


class Adversarial:
    """
    Utility class for generating Adversarial Sets for RTE.
    """
    def __init__(self, model_class, model_kwargs,
                 embedding_size=300, batch_size=1024, sequence_length=10,
                 entailment_idx=0, contradiction_idx=1, neutral_idx=2):
        self.model_class = model_class
        self.model_kwargs = model_kwargs

        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        self.entailment_idx = entailment_idx
        self.contradiction_idx = contradiction_idx
        self.neutral_idx = neutral_idx

        self.variable_name_to_variable = dict()

    def rule1(self):
        """
        Adversarial loss term computing (contradicts(S1, S2) - contradicts(S2, S1))^2,
        where the sentence embeddings S1 and S2 are selected adversarially.
        
        :return: tf.Tensor representing the adversarial loss.
        """
        sequence1 = tf.get_variable('rule1_sequence1',
                                    shape=[self.batch_size, self.sequence_length, self.embedding_size],
                                    initializer=tf.contrib.layers.xavier_initializer())
        self.variable_name_to_variable['rule1_sequence1'] = sequence1

        sequence2 = tf.get_variable('rule1_sequence2',
                                    shape=[self.batch_size, self.sequence_length, self.embedding_size],
                                    initializer=tf.contrib.layers.xavier_initializer())
        self.variable_name_to_variable['rule1_sequence2'] = sequence2

        a_model_kwargs = self.model_kwargs.copy()

        a_model_kwargs['sequence1'] = sequence1
        a_model_kwargs['sequence1_length'] = self.sequence_length

        a_model_kwargs['sequence2'] = sequence2
        a_model_kwargs['sequence2_length'] = self.sequence_length

        a_model = self.model_class(**a_model_kwargs)
        a_logits = a_model()

        p_s1_entails_s2 = tf.nn.softmax(a_logits)[:, self.contradiction_idx]

        b_model_kwargs = self.model_kwargs.copy()

        b_model_kwargs['sequence1'] = sequence2
        b_model_kwargs['sequence1_length'] = self.sequence_length

        b_model_kwargs['sequence2'] = sequence1
        b_model_kwargs['sequence2_length'] = self.sequence_length

        b_model = self.model_class(**b_model_kwargs)
        b_logits = b_model()

        p_s2_entails_s1 = tf.nn.softmax(b_logits)[:, self.contradiction_idx]

        return tf.nn.l2_loss(p_s1_entails_s2 - p_s2_entails_s1)

