from functools import partial
from typing import Callable, List, Tuple

import tensorflow as tf
import tensorflow_federated as tff

from ocddetection import data, losses, models
from ocddetection.learning.federated.stateless.impl.averaging import process


def __model_fn(
	window_size: int,
	hidden_size: int,
	pos_weight: float,
	metrics_fn: Callable[[], List[tf.keras.metrics.Metric]]
) -> tff.learning.Model:
	return tff.learning.from_keras_model(
		keras_model=models.bidirectional(window_size, len(data.SENSORS), hidden_size),
		loss=losses.WeightedBinaryCrossEntropy(pos_weight),
		input_spec=(
			tf.TensorSpec((None, window_size, len(data.SENSORS)), dtype=tf.float32),
			tf.TensorSpec((None, 1), dtype=tf.float32)
		),
		metrics=metrics_fn()
	)


def setup(
    window_size: int,
    hidden_size: int,
    pos_weight: float,
    training_metrics_fn: Callable[[], List[tf.metrics.Metric]],
    evaluation_metrics_fn: Callable[[], List[tf.metrics.Metric]],
    client_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
    server_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer]
) -> Tuple[Callable, Callable, Callable]:
	model_fn = partial(
		__model_fn,
		window_size=window_size,
		hidden_size=hidden_size,
		pos_weight=pos_weight,
		metrics_fn=training_metrics_fn
	)

	eval_model_fn = partial(
		__model_fn,
		window_size=window_size,
		hidden_size=hidden_size,
		pos_weight=pos_weight,
		metrics_fn=evaluation_metrics_fn
	)

	iterator = process.iterator(model_fn, server_optimizer_fn, client_optimizer_fn)
	validator = process.validator(model_fn)
	evaluator = process.evaluator(eval_model_fn)

	return iterator, validator, evaluator
