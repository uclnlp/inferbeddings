# -*- coding: utf-8 -*-

import tensorflow as tf

import inferbeddings.nli.util as util
from inferbeddings.nli import FeedForwardDAMP

from inferbeddings.nli.evaluation import accuracy

import logging

import pytest

logger = logging.getLogger(__name__)


@pytest.mark.light
def test_nli_damp():
    embedding_size = 300
    representation_size = 200
    max_len = None

    train_instances, dev_instances, test_instances = util.SNLI.generate()

    all_instances = train_instances + dev_instances + test_instances
    qs_tokenizer, a_tokenizer = util.train_tokenizer_on_instances(all_instances, num_words=None)

    vocab_size = qs_tokenizer.num_words if qs_tokenizer.num_words else max(qs_tokenizer.word_index.values()) + 1

    contradiction_idx = a_tokenizer.word_index['contradiction'] - 1
    entailment_idx = a_tokenizer.word_index['entailment'] - 1
    neutral_idx = a_tokenizer.word_index['neutral'] - 1

    train_dataset = util.to_dataset(train_instances, qs_tokenizer, a_tokenizer, max_len=max_len)
    dev_dataset = util.to_dataset(dev_instances, qs_tokenizer, a_tokenizer, max_len=max_len)
    test_dataset = util.to_dataset(test_instances, qs_tokenizer, a_tokenizer, max_len=max_len)

    sentence1_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence1')
    sentence2_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence2')

    sentence1_length_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence1_length')
    sentence2_length_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence2_length')

    label_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='label')
    dropout_keep_prob_ph = tf.placeholder(tf.float32, name='dropout_keep_prob')

    embedding_layer = tf.get_variable('embeddings', shape=[vocab_size, embedding_size])

    sentence1_embedding = tf.nn.embedding_lookup(embedding_layer, sentence1_ph)
    sentence2_embedding = tf.nn.embedding_lookup(embedding_layer, sentence2_ph)

    model_kwargs = dict(
        sequence1=sentence1_embedding, sequence1_length=sentence1_length_ph,
        sequence2=sentence2_embedding, sequence2_length=sentence2_length_ph,
        representation_size=representation_size, dropout_keep_prob=dropout_keep_prob_ph,
        use_masking=True)
    model_class = FeedForwardDAMP

    model = model_class(**model_kwargs)

    logits = model()
    predictions = tf.argmax(logits, axis=1, name='predictions')

    predictions_int = tf.cast(predictions, tf.int32)
    labels_int = tf.cast(label_ph, tf.int32)

    batch_size = 32

    restore_path = 'models/nli/damp_v1.ckpt'

    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, restore_path)

        dev_accuracy, _, _, _ = accuracy(session, dev_dataset, 'dev',
                                         sentence1_ph, sentence1_length_ph, sentence2_ph, sentence2_length_ph,
                                         label_ph, dropout_keep_prob_ph, predictions_int, labels_int,
                                         contradiction_idx, entailment_idx, neutral_idx, batch_size)

        test_accuracy, _, _, _ = accuracy(session, test_dataset, 'test',
                                          sentence1_ph, sentence1_length_ph, sentence2_ph, sentence2_length_ph,
                                          label_ph, dropout_keep_prob_ph, predictions_int, labels_int,
                                          contradiction_idx, entailment_idx, neutral_idx, batch_size)

        assert 0.86 < dev_accuracy < 0.89
        assert 0.86 < test_accuracy < 0.89

    tf.reset_default_graph()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    pytest.main([__file__])
