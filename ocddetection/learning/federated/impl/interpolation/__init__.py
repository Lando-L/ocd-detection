from functools import partial
from typing import Callable, Dict, List, Tuple

import tensorflow as tf
import tensorflow_federated as tff

from ocddetection import data, losses, models
from ocddetection.learning.federated.impl.interpolation import client, process


def __model_fn(
	window_size: int,
	hidden_size: int,
	dropout: float = 0.0,
	pos_weight: float = 1.0,
	metrics_fn: Callable[[], List[tf.keras.metrics.Metric]] = lambda: []
) -> tff.learning.Model:
	return tff.learning.from_keras_model(
		keras_model=models.bidirectional(window_size, len(data.SENSORS), hidden_size, dropout),
		loss=losses.WeightedBinaryCrossEntropy(pos_weight),
		input_spec=(
			tf.TensorSpec((None, window_size, len(data.SENSORS)), dtype=tf.float32),
			tf.TensorSpec((None, 1), dtype=tf.float32)
		),
		metrics=metrics_fn()
	)


def __coefficient_fn() -> tf.Variable:
	return tf.Variable(0.3, dtype=tf.float32, trainable=True)


def __client_state_fn(
	idx: int,
	pos_weight: float,
	weights: tff.learning.ModelWeights,
	mixing_coefficient: tf.Tensor
) -> client.State:
	return client.State(
		tf.constant(idx, dtype=tf.int32),
		tf.constant(pos_weight, dtype=tf.float32),
		weights,
		mixing_coefficient
	)


def setup(
	window_size: int,
	hidden_size: int,
	dropout: float,
	training_metrics_fn: Callable[[], List[tf.metrics.Metric]],
	evaluation_metrics_fn: Callable[[], List[tf.metrics.Metric]],
	client_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
	server_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer]
) -> Tuple[Dict[int, client.State], Callable, Callable, Callable]:
	model_fn = partial(
		__model_fn,
		window_size=window_size,
		hidden_size=hidden_size,
		dropout=dropout,
		metrics_fn=training_metrics_fn
	)

	eval_model_fn = partial(
		__model_fn,
		window_size=window_size,
		hidden_size=hidden_size,
		dropout=dropout,
		metrics_fn=evaluation_metrics_fn
	)

	weights = tff.learning.ModelWeights.from_model(model_fn())
	mixing_coefficient = __coefficient_fn().read_value()

	client_state_fn = partial(
		__client_state_fn,
		idx=-1,
		pos_weight=1.0,
		weights=weights,
		mixing_coefficient=mixing_coefficient
	)

	client_states = partial(
		__client_state_fn,
		weights=weights,
		mixing_coefficient=mixing_coefficient
	)

	iterator = process.iterator(
		__coefficient_fn,
		model_fn,
		client_state_fn,
		server_optimizer_fn,
		client_optimizer_fn
	)

	validator = process.validator(
		__coefficient_fn,
		model_fn,
		client_state_fn
	)

	evaluator = process.evaluator(
		__coefficient_fn,
		eval_model_fn,
		client_state_fn
	)

	return client_states, iterator, validator, evaluator


def create(
	window_size: int,
	hidden_size: int,
	optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
	metrics_fn: Callable[[], List[tf.keras.metrics.Metric]]
) -> Tuple[List, Callable[[], tff.learning.Model]]:
	model_fn = partial(
		__model_fn,
		window_size=window_size,
		hidden_size=hidden_size,
		dropout=0.0,
		pos_weight=1.0,
		metrics_fn=metrics_fn
	)

	client_states_fn = partial(
		__client_state_fn,
		weights=tff.learning.ModelWeights.from_model(model_fn()),
		mixing_coefficient=__coefficient_fn().read_value()
	)

	def client_model_fn(weights: tff.learning.ModelWeights, client_state: client.State) -> tff.learning.Model:
		model = model_fn()
		client.__mix_weights(client_state.mixing_coefficient, client_state.model, weights).assign_weights_to(model)

		return model

	return (
		process.__initialize_server(model_fn, optimizer_fn),
		client_states_fn,
		client_model_fn
	)
