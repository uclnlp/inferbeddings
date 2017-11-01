# -*- coding: utf-8 -*-

import collections
import tensorflow as tf


def _read_words(path):
    with tf.gfile.GFile(path, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def _file_to_word_ids(path, word_to_id):
    data = _read_words(path)
    return [word_to_id[word] for word in data if word in word_to_id]


def _build_vocab(path):
  data = _read_words(path)
  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  return word_to_id


def raw_data(train_path, valid_path=None, test_path=None):
    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id) if valid_path else None
    test_data = _file_to_word_ids(test_path, word_to_id) if test_path else None
    return train_data, valid_data, test_data, len(word_to_id)


def producer(raw_data, batch_size, num_steps, name=None):
    with tf.name_scope(name, "producer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0: batch_size * batch_len], [batch_size, batch_len])
        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(epoch_size, message="epoch_size == 0, decrease batch_size or num_steps")

        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps], [batch_size, (i + 1) * num_steps])

        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1], [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
    return x, y
