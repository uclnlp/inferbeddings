# -*- coding: utf-8 -*-

import numpy as np
from inferbeddings.visualization import hinton_diagram


def test_hinton_diagram():
    data = np.random.randn(10, 50)
    diagram = hinton_diagram(data)
    assert len(diagram) == 6967
