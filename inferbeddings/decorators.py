# -*- coding: utf-8 -*-

import os
import pickle
import atexit


def persist(path):
    def decorator(fun):
        cache = {}
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                cache = pickle.load(f)

        def write():
            with open(path, 'wb') as f:
                pickle.dump(cache, f)
        atexit.register(lambda: write())

        def new_f(*args):
            if tuple(args) not in cache:
                cache[tuple(args)] = fun(*args)
            return cache[args]
        return new_f
    return decorator
