# -*- coding: utf-8 -*-

import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


class Adversarial:
    def __init__(self, embeddings):
        self.embeddings = embeddings
