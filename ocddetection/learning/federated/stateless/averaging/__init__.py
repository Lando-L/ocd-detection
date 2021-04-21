from functools import partial
from typing import Callable, List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from ocddetection import data, losses, metrics, models
from ocddetection.learning.federated.stateless.averaging import process


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


def __server_optimizer_fn() -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.SGD(1.0, momentum=.9)


def __client_optimizer_fn(learning_rate: float) -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.SGD(learning_rate, momentum=.9)


def setup(
    window_size: int,
    batch_size: int,
    hidden_size: int,
    dropout_rate: float,
    learning_rate: float,
    pos_weight: float
) -> Tuple[Callable, Callable, Callable]:
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
    
    client_optimizer_fn = partial(__client_optimizer_fn, learning_rate=learning_rate)

    iterator = process.iterator(model_fn, __server_optimizer_fn, client_optimizer_fn)
    validator = process.validator(model_fn)
    evaluator = process.evaluator(eval_model_fn)

    return iterator, validator, evaluator
