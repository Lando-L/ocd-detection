from functools import partial
from ocddetection.types import ClientState
from typing import Callable, Dict, List, Tuple

import tensorflow as tf
import tensorflow_federated as tff

from ocddetection import data, losses, models
from ocddetection.learning.federated.stateful.impl.layers import client, process, utils


def __model_fn(
	window_size: int,
	hidden_size: int,
	pos_weight: float,
	metrics_fn: Callable[[], List[tf.keras.metrics.Metric]]
) -> utils.PersonalizationLayersDecorator:
	base, personalized, model = models.personalized_bidirectional(window_size, len(data.SENSORS), hidden_size, pos_weight)
	
	return utils.PersonalizationLayersDecorator(
		base,
		personalized,
		tff.learning.from_keras_model(
			model,
			loss=losses.WeightedBinaryCrossEntropy(pos_weight),
			input_spec=(
				tf.TensorSpec((None, window_size, len(data.SENSORS)), dtype=tf.float32),
				tf.TensorSpec((None, 1), dtype=tf.float32)
			),
			metrics=metrics_fn()
		)
	)


def __client_state_fn(idx: int, weights: tff.learning.ModelWeights):
	return client.State(tf.constant(idx), weights)


def setup(
  window_size: int,
	hidden_size: int,
	pos_weight: float,
	training_metrics_fn: Callable[[], List[tf.metrics.Metric]],
	evaluation_metrics_fn: Callable[[], List[tf.metrics.Metric]],
	client_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
	server_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer]
) -> Tuple[Dict[int, client.State], Callable, Callable, Callable]:
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

	weights = tff.learning.ModelWeights.from_model(model_fn().personalized_model)
	
	client_state_fn = partial(
		__client_state_fn,
		idx=-1,
		weights=weights
	)

	client_states = partial(
		__client_state_fn,
		weights=weights
	)

	iterator = process.iterator(
		model_fn,
		client_state_fn,
		server_optimizer_fn,
		client_optimizer_fn
	)

	validator = process.validator(
		model_fn,
		client_state_fn
	)

	evaluator = process.evaluator(
		eval_model_fn,
		client_state_fn
	)

	return client_states, iterator, validator, evaluator
	
