#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import MDS, TSNE
import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def score_TransE_L1(subject_embedding, predicate_embedding, object_embedding):
    translated_embedding = subject_embedding + predicate_embedding
    return - np.sum(np.abs(translated_embedding - object_embedding))


def score_TransE_L2(subject_embedding, predicate_embedding, object_embedding):
    translated_embedding = subject_embedding + predicate_embedding
    return - np.sqrt(np.sum(np.square(translated_embedding) - object_embedding))


def score_DistMult(subject_embedding, predicate_embedding, object_embedding):
    return np.sum(subject_embedding * predicate_embedding * object_embedding)


def score_ComplEx(subject_embedding, predicate_embedding, object_embedding):
    def dot3(a, b, c):
        return np.sum(a * b * c)
    n = subject_embedding.shape[0]
    s_re, s_im = subject_embedding[:n // 2], subject_embedding[n // 2:]
    p_re, p_im = predicate_embedding[:n // 2], predicate_embedding[n // 2:]
    o_re, o_im = object_embedding[:n // 2], object_embedding[n // 2:]
    return dot3(s_re, p_re, o_re) + dot3(s_re, p_im, o_im) + dot3(s_im, p_re, o_im) - dot3(s_im, p_im, o_re)


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Plot Embeddings', formatter_class=formatter)
    argparser.add_argument('model', action='store', type=str)
    argparser.add_argument('adversary', action='store', type=str)

    args = argparser.parse_args(argv)

    model_path = args.model
    adversary_path = args.adversary

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    with open(adversary_path, 'rb') as f:
        adversary_data = pickle.load(f)

    entity_embeddings = model_data['entities'][1:, :]
    predicate_embeddings = model_data['predicates'][1:, :]

    variables = adversary_data['variables']

    entity_to_index = model_data['entity_to_index']
    predicate_to_index = model_data['predicate_to_index']

    nb_entities = len(entity_to_index)

    for variable_name, embedding in variables.items():
        variable_name = variable_name.split('_')[2]
        entity_to_index[variable_name] = len(entity_to_index) + 1
        entity_embeddings = np.concatenate((entity_embeddings, embedding), axis=0)

    index_to_entity = {index: entity for entity, index in entity_to_index.items()}
    index_to_predicate = {index: predicate for predicate, index in predicate_to_index.items()}

    projector = MDS(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)

    entity_embeddings_proj = projector.fit_transform(entity_embeddings)

    plt.scatter(entity_embeddings_proj[:nb_entities, 0], entity_embeddings_proj[:nb_entities, 1], color='c')
    plt.scatter(entity_embeddings_proj[nb_entities:, 0], entity_embeddings_proj[nb_entities:, 1], color='r')

    for index, (x, y) in enumerate(zip(entity_embeddings_proj[:, 0], entity_embeddings_proj[:, 1]), 1):
        label = index_to_entity[index]
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

    plt.show()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
