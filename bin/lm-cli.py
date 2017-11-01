#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

import numpy as np
import tensorflow as tf

from inferbeddings.lm import reader
from inferbeddings.lm.model import LanguageModel


logging = tf.logging
flags = tf.flags

flags.DEFINE_integer("num_gpus", 1, "Number of GPUs.")
flags.DEFINE_integer("train_path", "data/lm/ptb/ptb.train.txt", "Training set.")
flags.DEFINE_integer("valid_path", "data/lm/ptb/ptb.valid.txt", "Validation set.")
flags.DEFINE_integer("test_path", "data/lm/ptb/ptb.test.txt", "Test set.")

FLAGS = flags.FLAGS
BASIC = "basic"
BLOCK = "block"


class Input(object):
  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.producer(data, batch_size, num_steps, name=name)


class SmallConfig(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


def get_config():
    config = SmallConfig()
    return config


def run_epoch(session, model, eval_op=None, verbose=False):
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size * max(1, FLAGS.num_gpus) /
                   (time.time() - start_time)))

    return np.exp(costs / iters)


def main(_):
    raw_data = reader.raw_data()
    train_data, valid_data, test_data, _ = raw_data

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size, eval_config.num_steps = 1, 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(- config.init_scale, config.init_scale)

        with tf.name_scope("train"):
            train_input = Input(config=config, data=train_data, name="train_input")
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = LanguageModel(is_training=True, config=config, input_=train_input)

        with tf.name_scope("valid"):
            valid_input = Input(config=config, data=valid_data, name="valid_input")
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                m = LanguageModel(is_training=False, config=config, input_=valid_input)

        with tf.name_scope("test"):
            valid_input = Input(config=config, data=test_data, name="test_input")
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                m = LanguageModel(is_training=False, config=eval_config, input_=valid_input)

if __name__ == '__main__':
    tf.app.run()
