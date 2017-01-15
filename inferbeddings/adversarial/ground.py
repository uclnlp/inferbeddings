# -*- coding: utf-8 -*-

import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


class GroundLoss:
    def __init__(self, clauses):
        self.clauses = clauses

