# -*- coding: utf-8 -*-

import pytest

import numpy as np

from inferbeddings.lm.beam import BeamSearch


def naive_predict(sample, state):
    return np.array(state)[None, :], state


@pytest.mark.light
def test_single_beam():
    prime_labels = [0, 1]
    initial_state = [0.1, 0.2, 0.3, 0.4, 0.5]

    bs = BeamSearch(naive_predict, initial_state, prime_labels)
    samples, scores = bs.search(None, None, k=1, maxsample=5)

    assert samples == [[0, 1, 4, 4, 4]]


@pytest.mark.light
def test_multiple_beams():
    prime_labels = [0, 1]
    initial_state = [0.1, 0.2, 0.3, 0.4, 0.5]

    bs = BeamSearch(naive_predict, initial_state, prime_labels)
    samples, scores = bs.search(None, None, k=4, maxsample=5)

    assert [0, 1, 4, 4, 4] in samples

    assert [0, 1, 4, 4, 3] in samples
    assert [0, 1, 4, 3, 4] in samples
    assert [0, 1, 3, 4, 4] in samples

    assert samples[np.argmin(scores)] == [0, 1, 4, 4, 4]

if __name__ == '__main__':
    pytest.main([__file__])
