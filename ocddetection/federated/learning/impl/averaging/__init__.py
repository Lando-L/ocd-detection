from functools import partial
from typing import Callable, Dict, Tuple

import tensorflow as tf
import tensorflow_federated as tff

from ocddetection import models
from ocddetection.data import preprocessing
from ocddetection.federated.learning.impl.personalization.interpolation import client, server, process


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


def __server_optimizer_fn() -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.SGD(1.0, momentum=.9)


def __client_optimizer_fn(learning_rate: float) -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.SGD(learning_rate, momentum=.9)


def setup(
    window_size: int,
    hidden_size: int,
    dropout_rate: float,
    learning_rate: float
) -> Tuple[Callable, Callable]:
    model_fn = partial(__model_fn, window_size=window_size, hidden_size=hidden_size, dropout_rate=dropout_rate)
    client_optimizer_fn = partial(__client_optimizer_fn, learning_rate=learning_rate)

    iterator = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn,
        __server_optimizer_fn
    )

    evaluator = tff.learning.build_federated_evaluation(
        model_fn
    )

    return iterator, evaluator
