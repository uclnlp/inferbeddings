# -*- coding: utf-8 -*-

import numpy as np
import inferbeddings.visualization.util as util


class HintonDiagram:
    def __init__(self, is_terminal=True):
        self.is_terminal = is_terminal

    def __call__(self, data):
        return util.hinton_diagram(data)


if __name__ == '__main__':
    rdata = np.random.randn(50, 100)
    hd = HintonDiagram(is_terminal=False)
    print(hd(rdata))
