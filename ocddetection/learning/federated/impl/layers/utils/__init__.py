import abc

import tensorflow as tf
import tensorflow_federated as tff


class ModelDecorator(tff.learning.Model, metaclass=abc.ABCMeta):
    def __init__(self, model: tff.learning.Model):
        self._model = model

    @property
    def trainable_variables(self):
        return self._model.trainable_variables
    
    @property
    def non_trainable_variables(self):
        return self._model.non_trainable_variables
    
    @property
    def local_variables(self):
        return self._model.local_variables
    
    @property
    def input_spec(self):
        return self._model.input_spec
    
    @tf.function
    def forward_pass(self, batch_input, training=True):
        return self._model.forward_pass(batch_input, training)
    
    @tf.function
    def report_local_outputs(self):
        return self._model.report_local_outputs()
    
    @property
    def federated_output_computation(self):
        return self._model.federated_output_computation


class PersonalizationLayersDecorator(ModelDecorator):
    def __init__(
        self,
        base: tf.keras.Model,
        personalized: tf.keras.Model,
        model: tff.learning.Model
    ):
        super(PersonalizationLayersDecorator, self).__init__(model)
        self._base = base
        self._personalized = personalized
    
    @property
    def base_model(self):
        return self._base
    
    @property
    def personalized_model(self):
        return self._personalized
