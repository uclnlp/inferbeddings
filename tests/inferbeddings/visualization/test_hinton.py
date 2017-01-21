# -*- coding: utf-8 -*-

import numpy as np
from inferbeddings.visualization.hinton import HintonDiagram


def test_hinton_diagram():
    data = np.random.randn(10, 50)
    hd = HintonDiagram(is_terminal=False)
    diagram = hd(data)
    assert len(diagram) == 6967
