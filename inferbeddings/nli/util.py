# -*- coding: utf-8 -*-

import gzip
import json

import numpy as np
import nltk

import logging

logger = logging.getLogger(__name__)


class SNLI:
    @staticmethod
    def to_instance(d, tokenize=None):
        sentence1 = d['sentence1']
        sentence1_parse = d['sentence1_parse']
        sentence1_tree = nltk.Tree.fromstring(sentence1_parse)
        sentence1_parse_tokens = sentence1_tree.leaves()
        sentence1_tokens = tokenize(sentence1) if tokenize else None

        sentence2 = d['sentence2']
        sentence2_parse = d['sentence2_parse']
        sentence2_tree = nltk.Tree.fromstring(sentence2_parse)
        sentence2_parse_tokens = sentence2_tree.leaves()
        sentence2_tokens = tokenize(sentence2) if tokenize else None

        gold_label = d['gold_label']

        instance = {
            'sentence1': sentence1,
            'sentence1_parse': sentence1_parse,
            'sentence1_parse_tokens': sentence1_parse_tokens,
            'sentence1_tokens': sentence1_tokens,

            'sentence2': sentence2,
            'sentence2_parse': sentence2_parse,
            'sentence2_parse_tokens': sentence2_parse_tokens,
            'sentence2_tokens': sentence2_tokens,

            'gold_label': gold_label
        }

        return instance

    @staticmethod
    def parse(path, tokenize=None, is_lower=False):
        res = None
        if path is not None:
            with gzip.open(path, 'rb') as f:
                res = []
                for line in f:
                    decoded_line = line.decode('utf-8')
                    if is_lower:
                        decoded_line = decoded_line.lower()
                    obj = json.loads(decoded_line)
                    instance = SNLI.to_instance(obj, tokenize=tokenize)
                    if instance['gold_label'] in {'entailment', 'neutral', 'contradiction'}:
                        res += [instance]
        return res

    @staticmethod
    def generate(train_path='data/snli/snli_1.0_train.jsonl.gz',
                 valid_path='data/snli/snli_1.0_dev.jsonl.gz',
                 test_path='data/snli/snli_1.0_test.jsonl.gz',
                 is_lower=False):

        tokenizer = nltk.tokenize.TreebankWordTokenizer()

        def tokenize(text):
            return tokenizer.tokenize(text)

        train_corpus = SNLI.parse(train_path, tokenize=tokenize, is_lower=is_lower)
        dev_corpus = SNLI.parse(valid_path, tokenize=tokenize, is_lower=is_lower)
        test_corpus = SNLI.parse(test_path, tokenize=tokenize, is_lower=is_lower)

        return train_corpus, dev_corpus, test_corpus


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
        model.sentence1: dataset['questions'],
        model.sentence2: dataset['supports'],
        model.sentence1_size: dataset['question_lengths'],
        model.sentence2_size: dataset['support_lengths'],
        model.label: dataset['answers']
    }


def instances_to_dataset(instances, token_to_index, label_to_index,
                         has_bos=False, has_eos=False, has_unk=False,
                         bos_idx=1, eos_idx=2, unk_idx=3,
                         max_len=None):
    assert (token_to_index is not None) and (label_to_index is not None)

    sentence1_idx, sentence2_idx, label_idx = [], [], []
    for instance in instances:
        _sentence1_idx, _sentence2_idx = [], []

        if has_bos:
            _sentence1_idx += [bos_idx]
            _sentence2_idx += [bos_idx]

        for token in instance['sentence1_parse_tokens']:
            if token in token_to_index:
                _sentence1_idx += [token_to_index[token]]
            elif has_unk:
                _sentence1_idx += [unk_idx]

        for token in instance['sentence2_parse_tokens']:
            if token in token_to_index:
                _sentence2_idx += [token_to_index[token]]
            elif has_unk:
                _sentence2_idx += [unk_idx]

        if has_eos:
            _sentence1_idx += [eos_idx]
            _sentence2_idx += [eos_idx]

        gold_label = instance['gold_label']
        assert gold_label in label_to_index

        sentence1_idx += [_sentence1_idx]
        sentence2_idx += [_sentence2_idx]
        label_idx += [label_to_index[gold_label]]

    assert len(sentence1_idx) == len(sentence2_idx) == len(label_idx)
    assert set(label_idx) == {0, 1, 2}

    sentence1_length = [len(s) for s in sentence1_idx]
    sentence2_length = [len(s) for s in sentence2_idx]

    ds = {
        'sentence1': pad_sequences(sentence1_idx, max_len=max_len),
        'sentence1_length': np.clip(a=np.array(sentence1_length), a_min=0, a_max=max_len),

        'sentence2': pad_sequences(sentence2_idx, max_len=max_len),
        'sentence2_length': np.clip(a=np.array(sentence2_length), a_min=0, a_max=max_len),

        'label': np.array(label_idx)
    }
    return ds
