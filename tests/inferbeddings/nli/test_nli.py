# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import inferbeddings.nli.util as util
from inferbeddings.nli import ConditionalBiLSTM, FeedForwardDAM, FeedForwardDAMP, ESIMv1

from inferbeddings.visualization import hinton_diagram

import logging

import pytest

logger = logging.getLogger(__name__)


def test_nli_damp():
    embedding_size = 300
    representation_size = 200
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
    sentence2_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentence2')

    sentence1_length_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence1_length')
    sentence2_length_ph = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence2_length')

    dropout_keep_prob_ph = tf.placeholder(tf.float32, name='dropout_keep_prob')

    embedding_layer = tf.get_variable('embeddings', shape=[vocab_size, embedding_size],
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      trainable=False)

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
    probabilities = tf.nn.softmax(logits)

    # {'entailment': '0.00833298', 'neutral': '0.00973773', 'contradiction': '0.981929'}
    # sentence1_str = '<bos> The boy is jumping <eos>'
    # sentence2_str = '<bos> The girl is jumping happily on the table <eos>'

    # {'entailment': '0.000107546', 'contradiction': '0.995034', 'neutral': '0.00485802'}
    # sentence1_str = '<bos> The girl is jumping happily on the table <eos>'
    # sentence2_str = '<bos> The boy is jumping <eos>'

    sentence1_str = '<bos> The boy is jumping happily on the table <eos>'
    sentence2_str = '<bos> The boy is jumping <eos>'

    sentence1_seq = [item for sublist in qs_tokenizer.texts_to_sequences([sentence1_str]) for item in sublist]
    sentence2_seq = [item for sublist in qs_tokenizer.texts_to_sequences([sentence2_str]) for item in sublist]

    restore_path = 'models/nli/damp_v1.ckpt'

    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, restore_path)

        feed_dict = {
            sentence1_ph: [sentence1_seq],
            sentence2_ph: [sentence2_seq],
            sentence1_length_ph: [len(sentence1_seq)],
            sentence2_length_ph: [len(sentence2_seq)],
            dropout_keep_prob_ph: 1.0
        }

        probabilities_value = session.run(probabilities, feed_dict=feed_dict)[0]

        answer = {
            'neutral': str(probabilities_value[neutral_idx]),
            'contradiction': str(probabilities_value[contradiction_idx]),
            'entailment': str(probabilities_value[entailment_idx])
        }

        print(answer)

        raw_attentions_value = session.run(model.raw_attentions, feed_dict=feed_dict)[0]

        print(hinton_diagram(raw_attentions_value))

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    pytest.main([__file__])
