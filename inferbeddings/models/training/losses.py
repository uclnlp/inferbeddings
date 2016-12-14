# -*- coding: utf-8 -*-

import tensorflow as tf
import sys


def get_function(function_name):
    this_module = sys.modules[__name__]
    if not hasattr(this_module, function_name):
        raise ValueError('Unknown objective function: {}'.format(function_name))
    return getattr(this_module, function_name)
