# -*- coding: utf-8 -*-

import inferbeddings.visualization.util as util


class HintonDiagram:
    def __init__(self, is_terminal=True):
        self.is_terminal = is_terminal

    def __call__(self, data):
        return util.hinton_diagram(data)
