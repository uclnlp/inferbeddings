from collections import namedtuple

import tensorflow as tf

from inferbeddings.knowledgebase import Fact
from inferbeddings.knowledgebase import KnowledgeBaseParser
from inferbeddings.models.training import constraints

ProvidedEmbeddings = namedtuple("ProvidedEmbeddings",
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


def pretrained_entity_embeddings(kb_parser: KnowledgeBaseParser, pretrained_embeddings_file):
    # construct a pre-trained matrix by loading the embeddings, turn into tf.constant
    with open(pretrained_embeddings_file, "r") as f:
        vocab, lookup = load_glove(f, kb_parser.entity_vocabulary)
        dim = lookup.shape[1]
        embedding_matrix = np.empty([len(kb_parser.entity_vocabulary) + 1, dim], dtype=np.float)
        for entity, index in kb_parser.entity_to_index.items():
            if entity in vocab:
                embedding_matrix[index, :] = lookup[vocab[entity], :]
            else:
                embedding_matrix[index, :] = lookup[vocab[entity], :]

        embedding_matrix_const = tf.constant(embedding_matrix)

    def entity_embeddings(nb_entities, entity_embeddings_size, entity_inputs):
        assert nb_entities == len(kb_parser.entity_vocabulary)
        assert entity_embeddings_size == dim
        embeddings = tf.nn.embedding_lookup(embedding_matrix_const, entity_inputs)
        return ProvidedEmbeddings(embeddings, (), (), embedding_matrix_const)

    return entity_embeddings


import numpy as np


def load_glove(stream, vocab):
    """Loads GloVe file and merges it if optional vocabulary
    Args:
        stream (iterable): An opened filestream to the GloVe file.
        vocab (dict=None): Word2idx dict of existing vocabulary.
    Returns:
        return_vocab (Vocabulary), lookup (matrix); Vocabulary contains the
                     word2idx and the matrix contains the embedded words.
    """
    print('[Loading GloVe]')
    word2idx = {}
    first_line = stream.readline()
    dim = len(first_line.split()) - 1
    lookup = np.empty([len(vocab) + 1, dim], dtype=np.float)
    lookup[0] = np.fromstring(first_line.split(maxsplit=1)[1], sep=' ')
    word2idx[first_line.split(maxsplit=1)[0]] = 0
    n = 1
    for line in stream:
        word, vec = line.rstrip().split(maxsplit=1)
        if vocab is None or word in vocab and word not in word2idx:
            # word = word.decode('utf-8')
            idx = len(word2idx)
            word2idx[word] = idx
            lookup[idx] = np.fromstring(vec, sep=' ')
        n += 1
        if n % 100000 == 0:
            print('  ' + str(n // 1000) + 'k vectors processed...\r')
    lookup.resize((len(word2idx), dim))
    return_vocab = word2idx
    print('[Loading GloVe DONE]')
    return return_vocab, lookup


if __name__ == "__main__":
    facts = [Fact("hypernym", ("animal", "human"))]
    parser = KnowledgeBaseParser(facts)
    provider = pretrained_entity_embeddings(parser, '/Users/riedel/projects/jtr/jtr/data/GloVe/glove.6B.50d.txt')
    provided = provider(2, 50, tf.constant([1],dtype=tf.int32))

    sess = tf.Session()
    print(sess.run(provided.embeddings))
    print(sess.run(provided.embedding_matrix))

    # with open('/Users/riedel/projects/jtr/jtr/data/GloVe/glove.6B.50d.txt', 'r') as f:
    #     vocab, lookup = load_glove(f, {"house"})
    #
    #     print(lookup)
