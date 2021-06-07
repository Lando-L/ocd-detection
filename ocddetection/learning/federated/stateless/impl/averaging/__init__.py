from functools import partial
from typing import Callable, List, Tuple

import tensorflow as tf
import tensorflow_federated as tff

from ocddetection import data, losses, metrics, models
from ocddetection.learning.federated.stateless.impl.averaging import process, server


def __model_fn(
	window_size: int,
	hidden_size: int,
	dropout: float,
	pos_weight: float,
	metrics_fn: Callable[[], List[tf.keras.metrics.Metric]]
) -> tff.learning.Model:
	return tff.learning.from_keras_model(
		keras_model=models.bidirectional(window_size, len(data.SENSORS), hidden_size, dropout, pos_weight),
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
		dropout: float,
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
		dropout=dropout,
		pos_weight=pos_weight,
		metrics_fn=training_metrics_fn
	)

	eval_model_fn = partial(
		__model_fn,
		window_size=window_size,
		hidden_size=hidden_size,
		dropout=dropout,
		pos_weight=pos_weight,
		metrics_fn=evaluation_metrics_fn
	)

	iterator = process.iterator(model_fn, server_optimizer_fn, client_optimizer_fn)
	validator = process.validator(model_fn)
	evaluator = process.evaluator(eval_model_fn)

	return iterator, validator, evaluator


def create(
	window_size: int,
	hidden_size: int,
	optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
	metrics_fn: Callable[[], List[tf.keras.metrics.Metric]]
) -> Tuple[server.State, Callable[[], tff.learning.Model]]:
	model_fn = partial(
		__model_fn,
		window_size=window_size,
		hidden_size=hidden_size,
		dropout=0.0,
		pos_weight=1.0,
		metrics_fn=metrics_fn
	)

	return (
		process.__initialize_server(model_fn, optimizer_fn),
		model_fn
	)
