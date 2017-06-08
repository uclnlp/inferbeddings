# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from inferbeddings.models.training.util import make_batches

import inferbeddings.nli.util as util
from inferbeddings.nli import ConditionalBiLSTM, FeedForwardDAM, FeedForwardDAMP, ESIMv1

import logging

import pytest

logger = logging.getLogger(__name__)


@pytest.mark.light
def test_nli_damp():
    embedding_size = 300
    representation_size = 200
    max_len = None

    train_instances, dev_instances, test_instances = util.SNLI.generate(bos='<bos>', eos='<eos>')

    all_instances = train_instances + dev_instances + test_instances
    qs_tokenizer, a_tokenizer = util.train_tokenizer_on_instances(all_instances, num_words=None)

    vocab_size = qs_tokenizer.num_words if qs_tokenizer.num_words else len(qs_tokenizer.word_index) + 1

    contradiction_idx = a_tokenizer.word_index['contradiction'] - 1
    entailment_idx = a_tokenizer.word_index['entailment'] - 1
    neutral_idx = a_tokenizer.word_index['neutral'] - 1

    dev_instances = dev_instances[:3]
    test_instances = test_instances[:3]

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
    probabilities = tf.nn.softmax(logits)
    predictions = tf.argmax(logits, axis=1, name='predictions')

    predictions_int = tf.cast(predictions, tf.int32)
    labels_int = tf.cast(label_ph, tf.int32)

    # {'neutral': '0.0318933', 'entailment': '0.964676', 'contradiction': '0.0034308'}
    # sentence1_str = '<bos> The boy is jumping <eos>'
    # sentence2_str = '<bos> The boy is jumping happily on the table <eos>'

    batch_size = 32
    restore_path = 'models/nli/damp_v1.ckpt'

    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, restore_path)

        def compute_accuracy(dataset, debug=False):
            nb_eval_instances = len(dataset['questions'])
            batches = make_batches(size=nb_eval_instances, batch_size=batch_size)

            p_vals, l_vals = [], []

            for batch_start, batch_end in batches:
                feed_dict = {
                    sentence1_ph: dataset['questions'][batch_start:batch_end],
                    sentence2_ph: dataset['supports'][batch_start:batch_end],
                    sentence1_length_ph: dataset['question_lengths'][batch_start:batch_end],
                    sentence2_length_ph: dataset['support_lengths'][batch_start:batch_end],
                    label_ph: dataset['answers'][batch_start:batch_end],
                    dropout_keep_prob_ph: 1.0
                }

                if debug:
                    print(feed_dict)
                    # print(session.run(sentence1_embedding[0], feed_dict=feed_dict))
                    # print(session.run(sentence2_embedding[0], feed_dict=feed_dict))
                    # print(session.run(model.transformed_sequence1[0], feed_dict=feed_dict))
                    # print(session.run(model.alpha[0], feed_dict=feed_dict))
                    # print(session.run(model.v1[0], feed_dict=feed_dict))
                    # print(session.run(logits, feed_dict=feed_dict))

                p_val, l_val = session.run([predictions_int, labels_int], feed_dict=feed_dict)

                p_vals += p_val.tolist()
                l_vals += l_val.tolist()

            matches = np.equal(p_vals, l_vals)
            return np.mean(matches)

        dev_accuracy = compute_accuracy(dev_dataset, debug=True)
        test_accuracy = compute_accuracy(test_dataset)

        print(dev_accuracy, test_accuracy)

    tf.reset_default_graph()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    pytest.main([__file__])
