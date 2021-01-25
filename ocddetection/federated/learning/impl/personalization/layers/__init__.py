from functools import partial
from typing import Callable, Dict, Tuple

import tensorflow as tf
import tensorflow_federated as tff

from ocddetection import models
from ocddetection.data import preprocessing
from ocddetection.federated.learning.impl.personalization.layers import client, process, utils


def __model_fn(window_size: int, hidden_size: int, dropout_rate: float) -> utils.PersonalizationLayersDecorator:
        base, personalized, model = models.personalized_bidirectional(
            window_size,
            len(preprocessing.SENSORS),
            len(preprocessing.LABEL2IDX),
            hidden_size,
            dropout_rate
        )
        
        return utils.PersonalizationLayersDecorator(
            base,
            personalized,
            tff.learning.from_keras_model(
                model,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                input_spec=(
                    tf.TensorSpec((None, window_size, len(preprocessing.SENSORS)), dtype=tf.float32),
                    tf.TensorSpec((None, window_size), dtype=tf.int32)
                ),
                metrics=[
                    tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
                ]
            )
        )


def __server_optimizer_fn() -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.SGD(1.0, momentum=.9)


def __client_optimizer_fn(learning_rate: float) -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.SGD(learning_rate)


def __client_state_fn(idx: int, weights: tff.learning.ModelWeights):
        return client.State(tf.constant(idx), weights)


def setup(
    window_size: int,
    hidden_size: int,
    dropout_rate: float,
    learning_rate: float,
    client_id2idx: Dict[int, int]
) -> Tuple[Dict[int, client.State], Callable, Callable]:
    model = __model_fn(window_size, hidden_size, dropout_rate)
    weights = tff.learning.ModelWeights.from_model(model.personalized_model)

    client_states = {
        i: __client_state_fn(idx, weights)
        for i, idx in client_id2idx.items()
    }

    model_fn = partial(__model_fn, window_size=window_size, hidden_size=hidden_size, dropout_rate=dropout_rate)
    client_state_fn = partial(__client_state_fn, idx=-1, weights=weights)
    client_optimizer_fn = partial(__client_optimizer_fn, learning_rate=learning_rate)

    iterator = process.iterator(
        model_fn,
        client_state_fn,
        __server_optimizer_fn,
        client_optimizer_fn
    )

    evaluator = process.evaluator(
        model_fn,
        client_state_fn
    )

    return client_states, iterator, evaluator
