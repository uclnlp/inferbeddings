# -*- coding: utf-8 -*-

from inferbeddings.rte.dam import AbstractDecomposableAttentionModel


class SimpleDAM(AbstractDecomposableAttentionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _transform_embeddings(self, embeddings, reuse=False):
        return embeddings

    def _transform_attend(self, sequence, reuse=False):
        return sequence

    def _transform_compare(self, sequence, reuse=False):
        return sequence
