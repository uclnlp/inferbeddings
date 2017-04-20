#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import tensorflow as tf

import jtr.jack.readers as readers
from jtr.jack.data_structures import load_labelled_data

from jtr.preprocess.vocab import Vocab

import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(os.path.basename(sys.argv[0]))

logger.info("Existing models:\n{}".format(", ".join(readers.readers.keys())))


# Create example reader with a basic config
embedding_dim = 128
config = {'batch_size': 128,
          'repr_dim': 128,
          'repr_dim_input': embedding_dim,
          'dropout': 0.1}

vocab = Vocab()

from jtr.jack.tasks.mcqa.simple_mcqa import SingleSupportFixedClassInputs, PairOfBiLSTMOverSupportAndQuestionModel, EmptyOutputModule
from jtr.jack.core import SharedVocabAndConfig, JTReader

shared_resources = SharedVocabAndConfig(vocab, config)
reader = JTReader(shared_resources,
                  SingleSupportFixedClassInputs(shared_resources),
                  PairOfBiLSTMOverSupportAndQuestionModel(shared_resources),
                  EmptyOutputModule())

print(vocab.sym2id)


# Loaded some test data to work on
# This loads train, dev, and test data of sizes (2k, 1k, 1k)
class TestDatasets(object):
    @staticmethod
    def generate_SNLI():
        snli_path, snli_data = 'SNLI/', []
        splits = ['train.json', 'dev.json', 'test.json']
        for split in splits:
            path = os.path.join(snli_path, split)
            snli_data.append(load_labelled_data(path))
        return snli_data


train, dev, test = TestDatasets.generate_SNLI()

print(len(train))
print(train[0])

# We creates hooks which keep track of the loss
# We also create 'the standard hook' for our model
from jtr.jack.train.hooks import LossHook
hooks = [LossHook(reader, iter_interval=10), readers.eval_hooks['snli_reader'](reader, dev, iter_interval=25)]


# Here we initialize our optimizer
# we choose Adam with standard momentum values and learning rate 0.001
learning_rate = 0.001
optim = tf.train.AdamOptimizer(learning_rate)

# Lets train the reader on the CPU for 2 epochs
reader.train(optim, train,
             hooks=hooks, max_epochs=1,
             device='/cpu:0')

print(vocab.sym2id)