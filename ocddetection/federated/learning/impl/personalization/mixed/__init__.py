from functools import partial
from typing import Callable, Dict, List, Tuple

import tensorflow as tf
import tensorflow_federated as tff

from ocddetection import models
from ocddetection.data import preprocessing
from ocddetection.federated.learning.impl.personalization.mixed import client, server, process


def __model_fn(window_size: int, hidden_size: int, dropout_rate: float) -> tff.learning.Model:     
    return tff.learning.from_keras_model(
        keras_model=models.bidirectional(
            window_size,
            len(preprocessing.SENSORS),
            len(preprocessing.LABEL2IDX),
            hidden_size,
            dropout_rate
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        input_spec=(
            tf.TensorSpec((None, window_size, len(preprocessing.SENSORS)), dtype=tf.float32),
            tf.TensorSpec((None, window_size), dtype=tf.int32)
        ),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
        ]
    )


def __coefficient_fn(size: int) -> tf.Variable:
    return [tf.Variable(0.3, dtype=tf.float32, trainable=True) for _ in range(size)]


def __server_optimizer_fn() -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.SGD(1.0, momentum=.9)


def __client_optimizer_fn(learning_rate: float) -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.SGD(learning_rate, momentum=.9)


def __client_state_fn(idx: int, weights: tff.learning.ModelWeights, mixing_coefficients: List[tf.Variable]) -> client.State:
        return client.State(tf.constant(idx), weights, mixing_coefficients)


def setup(
    window_size: int,
    hidden_size: int,
    dropout_rate: float,
    learning_rate: float,
    client_id2idx: Dict[int, int]
) -> Tuple[Dict[int, client.State], Callable, Callable]:
    model = __model_fn(window_size, hidden_size, dropout_rate)
    weights = tff.learning.ModelWeights.from_model(model)
    mixing_coefficients = __coefficient_fn(len(model.trainable_variables))

    client_states = {
        i: __client_state_fn(idx, weights, mixing_coefficients)
        for i, idx in client_id2idx.items()
    }

    coefficient_fn = partial(__coefficient_fn, size=len(model.trainable_variables))
    model_fn = partial(__model_fn, window_size=window_size, hidden_size=hidden_size, dropout_rate=dropout_rate)
    client_state_fn = partial(__client_state_fn, idx=-1, weights=weights, mixing_coefficients=mixing_coefficients)
    client_optimizer_fn = partial(__client_optimizer_fn, learning_rate=learning_rate)

    iterator = process.iterator(
        coefficient_fn,
        model_fn,
        client_state_fn,
        __server_optimizer_fn,
        client_optimizer_fn
    )

    evaluator = process.evaluator(
        coefficient_fn,
        model_fn,
        client_state_fn
    )

    return client_states, iterator, evaluator