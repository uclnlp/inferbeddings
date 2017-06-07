# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import inferbeddings.nli.util as util
from inferbeddings.nli import ConditionalBiLSTM, FeedForwardDAM, FeedForwardDAMP, ESIMv1
import logging

import pytest

logger = logging.getLogger(__name__)


def test_nli_damp():
    embedding_size = 300
    max_len = None

    train_instances, dev_instances, test_instances = util.SNLI.generate(bos='<BOS>', eos='<EOS>')

    all_instances = train_instances + dev_instances + test_instances
    qs_tokenizer, a_tokenizer = util.train_tokenizer_on_instances(all_instances, num_words=None)

    vocab_size = qs_tokenizer.num_words if qs_tokenizer.num_words else len(qs_tokenizer.word_index) + 1

    contradiction_idx = a_tokenizer.word_index['contradiction'] - 1
    entailment_idx = a_tokenizer.word_index['entailment'] - 1
    neutral_idx = a_tokenizer.word_index['neutral'] - 1

    train_dataset = util.to_dataset(train_instances, qs_tokenizer, a_tokenizer, max_len=max_len)
    dev_dataset = util.to_dataset(dev_instances, qs_tokenizer, a_tokenizer, max_len=max_len)
    test_dataset = util.to_dataset(test_instances, qs_tokenizer, a_tokenizer, max_len=max_len)

    sentence1_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence1')

