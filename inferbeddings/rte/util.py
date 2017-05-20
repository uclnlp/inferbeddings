# -*- coding: utf-8 -*-

import gzip
import json

import numpy as np
import tensorflow as tf

import tensorflow.contrib.keras as keras

import logging

logger = logging.getLogger(__name__)


class SNLI:
    @staticmethod
    def to_instance(d, prefix=None):
        _id, _support, _question, _answer = d['pairID'], d['sentence1'], d['sentence2'], d['gold_label']
        if prefix is not None:
            _support = '{} {}'.format(prefix, _support)
            _question = '{} {}'.format(prefix, _question)
        return {'id': _id, 'support': _support, 'question': _question, 'answer': _answer}

    @staticmethod
    def parse(path, prefix=None):
        res = None
        if path is not None:
            with gzip.open(path, 'rb') as f:
                res = []
                for line in f:
                    instance = SNLI.to_instance(json.loads(line.decode('utf-8')), prefix=prefix)
                    if instance['answer'] in {'entailment', 'neutral', 'contradiction'}:
                        res += [instance]
        return res

    @staticmethod
    def generate(train_path='data/snli/snli_1.0_train.jsonl.gz',
                 valid_path='data/snli/snli_1.0_dev.jsonl.gz',
                 test_path='data/snli/snli_1.0_test.jsonl.gz',
                 prefix=None):
        train_corpus = SNLI.parse(train_path, prefix=prefix)
        dev_corpus = SNLI.parse(valid_path, prefix=prefix)
        test_corpus = SNLI.parse(test_path, prefix=prefix)
        return train_corpus, dev_corpus, test_corpus


def count_trainable_parameters():
    """
    Count the number of trainable tensorflow parameters loaded in
    the current graph.
    """
    total_params = 0
    for variable in tf.trainable_variables():
        variable_params = np.prod([1] + [dim.value for dim in variable.get_shape()])
        logging.debug('{}: {} params'.format(variable.name, variable_params))
        total_params += variable_params
    return total_params


def pad_sequences(sequences, max_len=None, dtype='int32', padding='post', truncating='post', value=0.):
    """Pads each sequence to the same length (length of the longest sequence).

    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.

    Supports post-padding and pre-padding (default).

    # Arguments
        sequences: list of lists where each element is a sequence
        max_len: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            max_len either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.

    # Returns
        x: numpy array with dimensions (number_of_sequences, max_len)

    # Raises
        ValueError: in case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if max_len is None:
        max_len = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, max_len) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-max_len:]
        elif truncating == 'post':
            trunc = s[:max_len]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

def to_feed_dict(model, dataset):
    return {
        model.sentence1: dataset['questions'], model.sentence2: dataset['supports'],
        model.sentence1_size: dataset['question_lengths'], model.sentence2_size: dataset['support_lengths'],
        model.label: dataset['answers']}


def train_tokenizer_on_instances(instances, num_words=None):
    question_texts = [instance['question'] for instance in instances]
    support_texts = [instance['support'] for instance in instances]
    answer_texts = [instance['answer'] for instance in instances]

    qs_tokenizer = keras.preprocessing.text.Tokenizer(num_words=num_words)
    a_tokenizer = keras.preprocessing.text.Tokenizer()

    qs_tokenizer.fit_on_texts(question_texts + support_texts)
    a_tokenizer.fit_on_texts(answer_texts)
    return qs_tokenizer, a_tokenizer


def to_dataset(instances, qs_tokenizer, a_tokenizer, max_len=None, semi_sort=False):
    question_texts = [instance['question'] for instance in instances]
    support_texts = [instance['support'] for instance in instances]
    answer_texts = [instance['answer'] for instance in instances]

    assert qs_tokenizer is not None and a_tokenizer is not None

    questions = qs_tokenizer.texts_to_sequences(question_texts)
    supports = [s for s in qs_tokenizer.texts_to_sequences(support_texts)]
    answers = [answers - 1 for [answers] in a_tokenizer.texts_to_sequences(answer_texts)]

    """
    <<For efficient batching in TensorFlow, we semi-sorted the training data to first contain examples
    where both sentences had length less than 20, followed by those with length less than 50, and
    then the rest. This ensured that most training batches contained examples of similar length.>>

    -- https://arxiv.org/pdf/1606.01933.pdf
    """
    if semi_sort:
        triples_under_20, triples_under_50, triples_under_nfty = [], [], []
        for q, s, a in zip(questions, supports, answers):
            if len(q) < 20 and len(s) < 20:
                triples_under_20 += [(q, s, a)]
            elif len(q) < 50 and len(s) < 50:
                triples_under_50 += [(q, s, a)]
            else:
                triples_under_nfty += [(q, s, a)]
        questions, supports, answers = [], [], []
        for q, s, a in triples_under_20 + triples_under_50 + triples_under_nfty:
            questions += [q]
            supports += [s]
            answers += [a]

    question_lenths = [len(q) for q in questions]
    support_lenghs = [len(s) for s in supports]

    assert set(answers) == {0, 1, 2}

    return {
        'questions': pad_sequences(questions, max_len=max_len),
        'supports': pad_sequences(supports, max_len=max_len),
        'question_lengths': np.clip(a=np.array(question_lenths), a_min=0, a_max=max_len),
        'support_lengths': np.clip(a=np.array(support_lenghs), a_min=0, a_max=max_len),
        'answers': np.array(answers)
    }
