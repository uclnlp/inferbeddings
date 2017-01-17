# -*- coding: utf-8 -*-

import sys
import numpy as np
import logging

logger = logging.getLogger(__name__)


def sample_tuples(variables, entities, sample_size=1024, seed=None):
    rs = np.random.RandomState(0 if seed is None else seed)

    nb_entities, nb_variables = len(entities), len(variables)
    entities = np.array(entities)

    sample_size = min(nb_entities ** nb_variables, sample_size)

    tuple_set = set()
    while len(tuple_set) < sample_size:
        tuple_set |= {tuple(value for value in entities[rs.choice(nb_entities, nb_variables)])}

    def tuple_to_mapping(_tuple):
        return {var_name: var_value for var_name, var_value in zip(variables, _tuple)}

    return [tuple_to_mapping(_tuple) for _tuple in tuple_set]


def main(argv):
    mappings = sample_tuples(['a', 'b', 'c'], [1, 2, 3], sample_size=30)
    for mapping in mappings:
        print(mapping)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
