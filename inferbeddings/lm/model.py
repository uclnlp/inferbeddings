# -*- coding: utf-8 -*-

import tensorflow as tf


class LanguageModel(object):
    def __init__(self, is_training, config, input_):
        self._is_training = is_training
        self._input = input_
        self._rnn_params = None
        self._cell = None
        self.batch_size = input_.batch_size
        self.num_steps = input_.num_steps

        size = config.hidden_size
        vocab_size = config.vocab_size

        embedding = tf.get_variable("embedding", [vocab_size, size])
        inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        output, state = self._build_rnn_graph(inputs, config, is_training)

        softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

        # Use the contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(logits,
                                                input_.targets,
                                                tf.ones([self.batch_size, self.num_steps]),
                                                average_across_timesteps=False,
                                                average_across_batch=True)

        # Update the cost
        self._cost = tf.reduce_sum(loss)
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                   global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def _build_rnn_graph(self, inputs, config, is_training):
        return self._build_rnn_graph_lstm(inputs, config, is_training)

    def _get_lstm_cell(self, config, is_training):
        cell = tf.contrib.rnn.BasicLSTMCell(config.hidden_size,
                                            forget_bias=0.0,
                                            state_is_tuple=True,
                                            reuse=not is_training)
        return cell

    def _build_rnn_graph_lstm(self, inputs, config, is_training):
        cell = self._get_lstm_cell(config, is_training)
        if is_training and config.keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(config.num_layers)], state_is_tuple=True)
        self._initial_state = cell.zero_state(config.batch_size)
        state = self._initial_state
        outputs = []
        with tf.variable_scope("rnn"):
            for time_step in range(self.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                    (cell_output, state) = cell(inputs[:, time_step, :], state)
                    outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
        return output, state
