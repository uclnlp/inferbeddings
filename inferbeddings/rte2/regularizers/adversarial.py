# -*- coding: utf-8 -*-

import tensorflow as tf

class Adversarial:
    def __init__(self, model_class, model_kwargs,
                 entailment_idx=0, contradiction_idx=1, neutral_idx=2):
        self.model_class = model_class
        self.model_kwargs = model_kwargs

        self.entailment_idx = entailment_idx
        self.contradiction_idx = contradiction_idx
        self.neutral_idx = neutral_idx

        
