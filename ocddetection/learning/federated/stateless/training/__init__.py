from collections import namedtuple
from functools import partial, reduce
import os
from typing import Callable, Dict, Iterable, List, Text, Tuple

import matplotlib.pylab as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_federated as tff

from ocddetection import metrics
from ocddetection.data import preprocessing
from ocddetection.types import Metrics, ServerState, FederatedDataset
from ocddetection.learning.federated import common


Config = namedtuple(
  'Config',
  ['path', 'output', 'rounds', 'clients_per_round', 'checkpoint_rate', 'learning_rate', 'epochs', 'batch_size', 'window_size', 'pos_weight', 'hidden_size']
)


def __load_data(path, epochs, window_size, batch_size) -> Iterable[Tuple[FederatedDataset, FederatedDataset, FederatedDataset]]:
	files = pd.Series(
		[os.path.join(path, f'S{subject}-ADL{run}-AUGMENTED.csv') for subject in range(1, 5) for run in range(1, 6)],
		index=pd.MultiIndex.from_product([list(range(1, 5)), list(range(1, 6))]),
		name='path'
	)

	train_files, val_files, test_files = preprocessing.split(
		files,
		validation=[(subject, 4) for subject in range(1, 5)],
		test=[(subject, 5) for subject in range(1, 5)]
	)

	train = preprocessing.to_federated(train_files, epochs, window_size, batch_size)
	val = preprocessing.to_federated(val_files, 1, window_size, batch_size)
	test = preprocessing.to_federated(test_files, 1, window_size, batch_size)

	return train, val, test


def __training_metrics_fn() -> List[tf.keras.metrics.Metric]:
  return [
		metrics.AUC(from_logits=True, curve='PR', name='auc')
	]


def __validation_metrics_fn() -> List[tf.keras.metrics.Metric]:
  thresholds = list(np.linspace(0, 1, 200))
  return [
    metrics.Precision(from_logits=True, thresholds=thresholds, name='precision'),
    metrics.Recall(from_logits=True, thresholds=thresholds, name='recall')
  ]


def __client_optimizer_fn(learning_rate: float) -> tf.keras.optimizers.Optimizer:
  return tf.keras.optimizers.Adam(learning_rate)


def __server_optimizer_fn() -> tf.keras.optimizers.Optimizer:
  return tf.keras.optimizers.SGD(1.0, momentum=.9)


def __train_step(
	state: ServerState,
	dataset: FederatedDataset,
	clients_per_round: int,
	training_fn: Callable[[ServerState, List[tf.data.Dataset]], Tuple[ServerState, Metrics]]
) -> Tuple[ServerState, Metrics]:
	sampled_clients = common.sample_clients(dataset, clients_per_round)
	sampled_data = [dataset.data[client] for client in sampled_clients]

	next_state, metrics = training_fn(state, sampled_data)

	return next_state, metrics


def __validation_step(
	weights: tff.learning.ModelWeights,
	dataset: FederatedDataset,
	validation_fn: Callable[[tff.learning.ModelWeights, List[tf.data.Dataset]], Metrics]
) -> Metrics:
	return common.update_test_metrics(
		validation_fn(
			weights,
			[dataset.data[client] for client in dataset.clients]
		)
	)


def __fit(
	state: ServerState,
	round_num: int,
	checkpoint_rate: int,
	checkpoint_manager: tff.simulation.FileCheckpointManager,
	train_step_fn: Callable[[ServerState], Tuple[ServerState, Metrics]],
	validation_step_fn: Callable[[tff.learning.ModelWeights], Metrics]
) -> ServerState:
	next_state, metrics = train_step_fn(state)
	mlflow.log_metrics(metrics, step=round_num)

	if round_num % checkpoint_rate == 0:
		test_metrics = validation_step_fn(next_state.model)
		mlflow.log_metrics(test_metrics, step=round_num)
		checkpoint_manager.save_checkpoint(next_state, round_num)

	return next_state


def __evaluate(
	weights: tff.learning.ModelWeights,
	dataset: FederatedDataset,
	evaluation_fn: Callable[[tff.learning.ModelWeights, List[tf.data.Dataset]], Tuple[tf.Tensor, Dict[Text, tf.Tensor]]]
) -> None:
	confusion_matrix, metrics = evaluation_fn(
		weights,
		[dataset.data[client] for client in dataset.clients]
	)

	# Confusion Matrix
	fig, ax = plt.subplots(figsize=(16, 8))

	sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap=sns.color_palette("Blues"), ax=ax)
	
	ax.set_xlabel('Predicted')
	ax.set_ylabel('Ground Truth')

	mlflow.log_figure(fig, f'confusion_matrix.png')
	plt.close(fig)

	# Precision Recall
	fig, ax = plt.subplots(figsize=(16, 8))

	sns.lineplot(x=metrics['recall'], y=metrics['precision'], ax=ax)

	ax.set_xlabel('Recall')
	ax.set_xlim(0., 1.)

	ax.set_ylabel('Precision')
	ax.set_ylim(0., 1.)

	mlflow.log_figure(fig, f'precision_recall.png')
	plt.close(fig)


def run(
	experiment_name: str,
	run_name: str,
	setup_fn: Callable[[int, int, float, Callable, Callable, Callable, Callable], Tuple[Callable, Callable, Callable]],
	config: Config
) -> None:
	mlflow.set_experiment(experiment_name)
	
	train, val, _ = __load_data(config.path, config.epochs, config.window_size, config.batch_size)

	checkpoint_manager = tff.simulation.FileCheckpointManager(config.output)

	client_optimizer_fn = partial(
		__client_optimizer_fn,
		learning_rate=config.learning_rate
	)

	iterator, validator, evaluator = setup_fn(
		config.window_size,
		config.hidden_size,
		config.pos_weight,
		__training_metrics_fn,
		__validation_metrics_fn,
		client_optimizer_fn,
		__server_optimizer_fn
	)

	train_step = partial(
		__train_step,
		dataset=train,
		clients_per_round=config.clients_per_round,
		training_fn=iterator.next
	)

	validation_step = partial(
		__validation_step,
		dataset=val,
		validation_fn=validator
	)

	fitting_fn = partial(
		__fit,
		checkpoint_rate=config.checkpoint_rate,
		checkpoint_manager=checkpoint_manager,
		train_step_fn=train_step,
		validation_step_fn=validation_step
	)

	evaluation_fn = partial(
		__evaluate,
		dataset=val,
		evaluation_fn=evaluator
	)

	with mlflow.start_run(run_name=run_name):
		mlflow.log_params(config._asdict())

		state = reduce(
			fitting_fn,
			range(1, config.rounds + 1),
			iterator.initialize()
		)

		evaluation_fn(state.model)
