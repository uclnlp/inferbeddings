# -*- coding: utf-8 -*-

import numpy as np
from colorclass import Color
from terminaltables import SingleTable

def hinton_diagram(arr, max_arr=None):
    max_arr = arr if max_arr is None else max_arr
    max_val = max(abs(np.max(max_arr)), abs(np.min(max_arr)))
    diagram = [list([hinton_diagram_value(x, max_val) for x in _arr]) for _arr in arr]

    table = SingleTable(diagram)
    table.inner_heading_row_border = False
    table.inner_footing_row_border = False
    table.inner_column_border = False
    table.inner_row_border = False
    table.column_max_width = 1

    return table.table


def hinton_diagram_value(val, max_val):
    chars = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
    if abs(abs(val) - max_val) < 1e-8:
        step = len(chars) - 1
    else:
        step = int(abs(float(val) / max_val) * len(chars))
    attr = 'red' if val < 0 else 'green'
    return Color('{auto' + attr + '}' + str(chars[step]) + '{/auto' + attr + '}')
