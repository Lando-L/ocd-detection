from collections import namedtuple

import tensorflow as tf


ModelWeights = collections.namedtuple('ModelWeights', ['trainable', 'non_trainable'])
ModelOutputs = collections.namedtuple('ModelOutputs', 'loss')

class Wrapper(object):
    def __init__(self, model, input_spec, loss_fn):
        super(Wrapper, self).__init__()
        self.model = model
        self.input_spec = input_spec
        self.loss_fn = loss_fn

    def call(self, X, y, training=True):
        y_hat, _ = model()
