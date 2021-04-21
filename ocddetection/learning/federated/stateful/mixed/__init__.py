from functools import partial
from typing import Callable, Dict, List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from ocddetection import data, losses, metrics, models
from ocddetection.learning.federated.stateful.mixed import client, server, process


def __model_fn(
    window_size: int,
    hidden_size: int,
    dropout_rate: float,
    pos_weight: float,
    metrics_fn: Callable[[], List[tf.keras.metrics.Metric]]
) -> tff.learning.Model:
    return tff.learning.from_keras_model(
        keras_model=models.bidirectional(
            window_size,
            len(data.SENSORS),
            hidden_size,
            dropout_rate
        ),
        loss=losses.WeightedBinaryCrossEntropy(pos_weight),
        input_spec=(
            tf.TensorSpec((None, window_size, len(data.SENSORS)), dtype=tf.float32),
            tf.TensorSpec((None, 1), dtype=tf.float32)
        ),
        metrics=metrics_fn()
    )


def __training_metrics_fn() -> List[tf.keras.metrics.Metric]:
    return [
        metrics.AUC(from_logits=True, curve='PR', name='pr_auc')
    ]


def __evaluation_metrics_fn() -> List[tf.keras.metrics.Metric]:
    thresholds = list(np.linspace(0, 1, 200))
    return [
        metrics.Precision(from_logits=True, thresholds=thresholds, name='precision'),
        metrics.Recall(from_logits=True, thresholds=thresholds, name='recall')
    ]


def __coefficient_fn(size: int) -> tf.Variable:
    return [tf.Variable(0.3, dtype=tf.float32, trainable=True) for _ in range(size)]


def __server_optimizer_fn() -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.SGD(1.0, momentum=.9)


def __client_optimizer_fn(learning_rate: float) -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.SGD(learning_rate, momentum=.9)


def __client_state_fn(idx: int, weights: tff.learning.ModelWeights, mixing_coefficients: List[tf.Tesnor]) -> client.State:
    return client.State(tf.constant(idx), weights, mixing_coefficients)


def setup(
    window_size: int,
    batch_size: int,
    hidden_size: int,
    dropout_rate: float,
    learning_rate: float,
    pos_weight: float,
    client_id2idx: Dict[int, int]
) -> Tuple[Dict[int, client.State], Callable, Callable, Callable]:
    model_fn = partial(
        __model_fn,
        window_size=window_size,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        pos_weight=pos_weight,
        metrics_fn=__training_metrics_fn
    )

    eval_model_fn = partial(
        __model_fn,
        window_size=window_size,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        pos_weight=pos_weight,
        metrics_fn=__evaluation_metrics_fn
    )

    coefficient_fn = partial(
        __coefficient_fn,
        size=len(model.trainable_variables)
    )

    model = model_fn()
    weights = tff.learning.ModelWeights.from_model(model)
    mixing_coefficients = coefficient_fn()

    client_states = {
        i: __client_state_fn(idx, weights, [v.read_value() for v in mixing_coefficients])
        for i, idx in client_id2idx.items()
    }

    client_state_fn = partial(
        __client_state_fn,
        idx=-1,
        weights=weights,
        mixing_coefficients=mixing_coefficients
    )

    client_optimizer_fn = partial(
        __client_optimizer_fn,
        learning_rate=learning_rate
    )

    iterator = process.iterator(
        coefficient_fn,
        model_fn,
        client_state_fn,
        __server_optimizer_fn,
        client_optimizer_fn
    )

    validator = process.validator(
        coefficient_fn,
        model_fn,
        client_state_fn
    )

    evaluator = process.evaluator(
        coefficient_fn,
        model_fn,
        client_state_fn
    )

    return client_states, iterator, validator, evaluator
