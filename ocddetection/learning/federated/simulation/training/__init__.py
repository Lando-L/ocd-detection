from collections import namedtuple
from functools import partial, reduce
import os
from typing import Callable, Dict, List, Text, Tuple

import matplotlib.pylab as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_federated as tff

from ocddetection import metrics
from ocddetection.data import preprocessing
from ocddetection.types import Metrics, ServerState, ClientState, FederatedDataset
from ocddetection.learning.federated.simulation import common


Config = namedtuple(
  'Config',
  ['path', 'output', 'rounds', 'clients_per_round', 'checkpoint_rate', 'learning_rate', 'epochs', 'batch_size', 'window_size', 'pos_weights', 'hidden_size', 'dropout']
)


def __load_data(path, epochs, window_size, batch_size) -> Tuple[FederatedDataset, FederatedDataset, FederatedDataset]:
	files = pd.Series(
		[os.path.join(path, f'S{subject}-ADL{run}-AUGMENTED.csv') for subject in range(1, 5) for run in range(1, 6)],
		index=pd.MultiIndex.from_product([list(range(1, 5)), list(range(1, 6))]),
		name='path'
	)

	train_files, val_files, test_files = preprocessing.split(
		files,
		validation=[],
		test=[(subject, 5) for subject in range(1, 5)]
	)

	train = preprocessing.to_federated(train_files, epochs, window_size, batch_size)
	val = preprocessing.to_federated(val_files, 1, window_size, batch_size)
	test = preprocessing.to_federated(test_files, 1, window_size, batch_size)

	return train, val, test


def __training_metrics_fn() -> List[tf.keras.metrics.Metric]:
  return [
		metrics.SigmoidDecorator(tf.keras.metrics.AUC(curve='PR'), name='auc')
	]


def __validation_metrics_fn() -> List[tf.keras.metrics.Metric]:
	thresholds = list(np.linspace(0, 1, 200, endpoint=False))
	return [
		metrics.SigmoidDecorator(tf.keras.metrics.AUC(curve='PR'), name='auc'),
		metrics.SigmoidDecorator(tf.keras.metrics.Precision(thresholds=thresholds), name='precision'),
		metrics.SigmoidDecorator(tf.keras.metrics.Recall(thresholds=thresholds), name='recall'),
		metrics.SigmoidDecorator(tf.keras.metrics.BinaryAccuracy(), name='accuracy'),
  ]


def __client_optimizer_fn(learning_rate: float) -> tf.keras.optimizers.Optimizer:
  return tf.keras.optimizers.SGD(learning_rate)


def __server_optimizer_fn() -> tf.keras.optimizers.Optimizer:
  return tf.keras.optimizers.SGD(1.0, momentum=0.9)


def __train_step(
  server_state: ServerState,
  client_states: Dict[int, ClientState],
  dataset: FederatedDataset,
  clients_per_round: int,
  client_idx2id: Dict[int, int],
  training_fn: Callable[[ServerState, List[tf.data.Dataset], List[ClientState]], Tuple[ServerState, Metrics, List[ClientState]]]
) -> Tuple[ServerState, Metrics, Dict[int, ClientState]]:
  sampled_clients = common.sample_clients(dataset, clients_per_round)
  sampled_data = [dataset.data[client] for client in sampled_clients]
  sampled_client_states = [client_states[client] for client in sampled_clients]
  
  next_state, metrics, updated_client_states = training_fn(server_state, sampled_data, sampled_client_states)

  next_client_states = {
    **client_states,
    **{
      client_idx2id[client.client_index.numpy()]: client
      for client in updated_client_states
    }
  }

  return next_state, metrics, next_client_states


def __validation_step(
	weights: tff.learning.ModelWeights,
	client_states: Dict[int, ClientState],
	dataset: FederatedDataset,
	validation_fn: Callable[[tff.learning.ModelWeights, List[tf.data.Dataset], List[ClientState]], Metrics]
) -> Metrics:
	return common.update_test_metrics(
		validation_fn(
			weights,
			[dataset.data[client] for client in dataset.clients],
			[client_states[client] for client in dataset.clients]
		)
	)


def __fit(
	state: Tuple[ServerState, Dict[int, ClientState]],
	round_num: int,
	checkpoint_rate: int,
	checkpoint_manager: tff.simulation.FileCheckpointManager,
	train_step_fn: Callable[[ServerState, Dict[int, ClientState]], Tuple[ServerState, Metrics, Dict[int, ClientState]]],
	validation_step_fn: Callable[[tff.learning.ModelWeights, Dict[int, ClientState]], Metrics]
) -> Tuple[ServerState, Dict[int, ClientState]]:
	next_state, metrics, next_client_states = train_step_fn(state[0], state[1])
	mlflow.log_metrics(metrics, step=round_num)

	test_metrics = validation_step_fn(next_state.model, next_client_states)
	mlflow.log_metrics(test_metrics, step=round_num)

	if round_num % checkpoint_rate == 0:
		checkpoint_manager.save_checkpoint([next_state, next_client_states], round_num)

	return next_state, next_client_states


def __evaluate(
	weights: tff.learning.ModelWeights,
  client_states: Dict[int, ClientState],
	dataset: FederatedDataset,
	evaluation_fn: Callable[[tff.learning.ModelWeights, List[tf.data.Dataset], List[ClientState]], Tuple[tf.Tensor, Dict[Text, tf.Tensor], Dict[Text, tf.Tensor]]]
) -> None:
	confusion_matrix, aggregated_metrics, client_metrics = evaluation_fn(
		weights,
		[dataset.data[client] for client in dataset.clients],
    [client_states[client] for client in dataset.clients]
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

	sns.lineplot(x=aggregated_metrics['recall'], y=aggregated_metrics['precision'], ax=ax)

	ax.set_xlabel('Recall')
	ax.set_xlim(0., 1.)

	ax.set_ylabel('Precision')
	ax.set_ylim(0., 1.)

	mlflow.log_figure(fig, f'precision_recall.png')
	plt.close(fig)

	# Client Metrics
	auc = metrics.SigmoidDecorator(tf.keras.metrics.AUC(curve='PR'), name='auc')
	accuracy = metrics.SigmoidDecorator(tf.keras.metrics.BinaryAccuracy(), name='accuracy')

	for client, metric in zip(client_states.keys(), iter(client_metrics)):
		tf.nest.map_structure(lambda v, t: v.assign(t), auc.variables, list(metric['auc']))
		tf.nest.map_structure(lambda v, t: v.assign(t), accuracy.variables, list(metric['accuracy']))

		mlflow.log_metric(f'client_{client}_val_auc', auc.result().numpy())
		mlflow.log_metric(f'client_{client}_val_acc', accuracy.result().numpy())


def run(
	experiment_name: str,
	run_name: str,
	setup_fn: Callable[[int, int, float, Dict[int, int], Callable, Callable, Callable, Callable], Tuple[Dict[int, ClientState], Callable, Callable, Callable]],
	config: Config
) -> None:
	mlflow.set_experiment(experiment_name)

	client_state_fn, iterator, validator, evaluator = setup_fn(
    config.window_size,
    config.hidden_size,
		config.dropout,
    __training_metrics_fn,
    __validation_metrics_fn,
    partial(__client_optimizer_fn, learning_rate=config.learning_rate),
    __server_optimizer_fn
	)
	
	train, _, test = __load_data(config.path, config.epochs, config.window_size, config.batch_size)

	checkpoint_manager = tff.simulation.FileCheckpointManager(config.output)

	client_idx2id = list(sorted(train.clients.union(test.clients)))
	client_states = {
		i: client_state_fn(idx, pos_weight)
		for idx, (i, pos_weight) in enumerate(zip(client_idx2id, config.pos_weights))
	}

	train_step = partial(
		__train_step,
		dataset=train,
    clients_per_round=config.clients_per_round,
    client_idx2id=client_idx2id,
    training_fn=iterator.next
	)

	validation_step = partial(
		__validation_step,
		dataset=test,
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
		dataset=test,
		evaluation_fn=evaluator
	)

	with mlflow.start_run(run_name=run_name):
		mlflow.log_params(config._asdict())

		server_state, client_states = reduce(
			fitting_fn,
			range(1, config.rounds + 1),
			(iterator.initialize(), client_states)
		)

		evaluation_fn(server_state.model, client_states)
