#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import MDS, TSNE
from inferbeddings.parse import parse_clause
from inferbeddings.adversarial.ground import GroundLoss

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


# Triple/Fact Scoring Functions
def score_TransE_L1(subject_embedding, predicate_embedding, object_embedding):
    translated_embedding = subject_embedding + predicate_embedding
    return - np.sum(np.abs(translated_embedding - object_embedding))


def score_TransE_L2(subject_embedding, predicate_embedding, object_embedding):
    translated_embedding = subject_embedding + predicate_embedding
    return - np.sqrt(np.sum(np.square(translated_embedding) - object_embedding))


def dot3(a, b, c):
    return np.sum(a * b * c)


def score_DistMult(subject_embedding, predicate_embedding, object_embedding):
    return dot3(subject_embedding, predicate_embedding, object_embedding)


def score_ComplEx(subject_embedding, predicate_embedding, object_embedding):
    n = subject_embedding.shape[0]
    s_re, s_im = subject_embedding[:n // 2], subject_embedding[n // 2:]
    p_re, p_im = predicate_embedding[:n // 2], predicate_embedding[n // 2:]
    o_re, o_im = object_embedding[:n // 2], object_embedding[n // 2:]
    return dot3(s_re, p_re, o_re) + dot3(s_re, p_im, o_im) + dot3(s_im, p_re, o_im) - dot3(s_im, p_im, o_re)


# Clause Scoring Functions
def score_atom(atom, feed_dict,
               entity_to_index, entity_embeddings,
               predicate_to_index, predicate_embeddings,
               scoring_function):
    arg1_name, arg2_name, predicate_name = atom.arguments[0].name, atom.arguments[1].name, atom.predicate.name
    s_idx, o_idx, p_idx = feed_dict[arg1_name], feed_dict[arg2_name], predicate_to_index[predicate_name]
    score_value = scoring_function(entity_embeddings[s_idx - 1, :], predicate_embeddings[p_idx - 1, :], entity_embeddings[o_idx - 1, :])
    return score_value


def score_conjunction(atoms, feed_dict, *args, **kwargs):
    atom_scores = [score_atom(atom, feed_dict, *args, **kwargs) for atom in atoms]
    return min(atom_scores)


def loss_clause(clause, feed_dict, *args, **kwargs):
    head, body = clause.head, clause.body
    score_head = score_atom(head, feed_dict, *args, **kwargs)
    score_body = score_conjunction(body, feed_dict, *args, **kwargs)
    return np.max([0, score_body - score_head])


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Plot Embeddings', formatter_class=formatter)
    argparser.add_argument('model', action='store', type=str)
    argparser.add_argument('adversary', action='store', type=str)
    argparser.add_argument('--clauses', '-c', action='store', type=str, default=None)

    args = argparser.parse_args(argv)

    model_path = args.model
    adversary_path = args.adversary
    clauses_path = args.clauses

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    with open(adversary_path, 'rb') as f:
        adversary_data = pickle.load(f)

    entity_embeddings = model_data['entities'][1:, :]
    predicate_embeddings = model_data['predicates'][1:, :]

    variables = adversary_data['variables']

    entity_to_index = model_data['entity_to_index']
    predicate_to_index = model_data['predicate_to_index']

    entity_indices = sorted(set(entity_to_index.values()))
    predicate_indices = sorted(set(predicate_to_index.values()))

    clauses = None
    if clauses_path is not None:
        with open(clauses_path, 'r') as f:
            clauses = [parse_clause(line.strip()) for line in f.readlines()]
        clause_to_variable_names = {clause: GroundLoss.get_variable_names(clause) for clause in clauses}
        clause_to_mappings = {clause: GroundLoss.sample_mappings(clause_to_variable_names[clause], entity_indices)
                              for clause in clauses}

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

    # Finding the maximum violators
    if clauses is not None:
        kwargs = {
            'entity_to_index': entity_to_index,
            'entity_embeddings': entity_embeddings,
            'predicate_to_index': predicate_to_index,
            'predicate_embeddings': predicate_embeddings,
            'scoring_function': score_TransE_L1
        }

        for clause, mappings in clause_to_mappings.items():
            mapping_loss_lst = [(mapping, loss_clause(clause, mapping, **kwargs)) for mapping in mappings]
            # Find the most violating variable assignment (i.e. the variable assignment with the highest loss)
            import operator
            most_violating_mapping = max(mapping_loss_lst, key=operator.itemgetter(1))[0]

            logger.info('Most violating mapping: {}'.format(most_violating_mapping))
            for variable_name, entity_idx in most_violating_mapping.items():
                plt.scatter(entity_embeddings_proj[entity_idx, 0], entity_embeddings_proj[entity_idx, 1], color='b')

    for index, (x, y) in enumerate(zip(entity_embeddings_proj[:, 0], entity_embeddings_proj[:, 1]), 1):
        label = index_to_entity[index]
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

    plt.show()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
