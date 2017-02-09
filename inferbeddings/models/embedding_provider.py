from typing import NamedTuple

import tensorflow as tf

from inferbeddings.models.training import constraints

ProvidedEmbeddings = NamedTuple("ProvidedEmbeddings",
                                ('embeddings', 'trainable_variables', 'projection_steps', 'embedding_matrix'))


def default_entity_embeddings(nb_entities, entity_embedding_size, entity_inputs):
    entity_embedding_layer = tf.get_variable('entities', shape=[nb_entities + 1, entity_embedding_size],
                                             initializer=tf.contrib.layers.xavier_initializer())

    entity_embeddings = tf.nn.embedding_lookup(entity_embedding_layer, entity_inputs)
    return ProvidedEmbeddings(entity_embeddings, [entity_embedding_layer],
                              (constraints.renorm_update(entity_embedding_layer, norm=1.0),), entity_embedding_layer)


def default_predicate_embeddings(nb_predicates, predicate_embedding_size, walk_inputs):
    predicate_embedding_layer = tf.get_variable('predicates', shape=[nb_predicates + 1, predicate_embedding_size],
                                                initializer=tf.contrib.layers.xavier_initializer())
    predicate_embeddings = tf.nn.embedding_lookup(predicate_embedding_layer, walk_inputs)
    return ProvidedEmbeddings(predicate_embeddings, (predicate_embedding_layer,), [], predicate_embedding_layer)


def pretrained_entity_embeddings(kb_parser, pretrained_embeddings_file):
    # construct a pre-trained matrix by loading the embeddings, turn into tf.constant

    def entity_embeddings(nb_entities, entity_embeddings_size, entity_inputs):
        return None

    return entity_embeddings
