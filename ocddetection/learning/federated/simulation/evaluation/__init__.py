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
from ocddetection.types import Metrics, ServerState, ClientState, FederatedDataset
from ocddetection.learning.federated.simulation import common


Config = namedtuple(
  'Config',
  ['path', 'rounds', 'clients_per_round', 'learning_rate', 'epochs', 'batch_size', 'window_size', 'pos_weight', 'hidden_size', 'dropout']
)


def __load_data(path, epochs, window_size, batch_size) -> Iterable[Tuple[FederatedDataset, FederatedDataset, FederatedDataset]]:
  for i in range(1, 5):
    files = pd.Series(
      [os.path.join(path, f'S{subject}-ADL{run}-AUGMENTED.csv') for subject in range(1, 5) for run in range(1, 6)],
      index=pd.MultiIndex.from_product([list(range(1, 5)), list(range(1, 6))]),
      name='path'
    )

    train_files, val_files, test_files = preprocessing.split(
      files,
      validation=[(subject, i) for subject in range(1, 5)],
      test=[(subject, 5) for subject in range(1, 5)]
    )

    train = preprocessing.to_federated(train_files, epochs, window_size, batch_size)
    val = preprocessing.to_federated(val_files, 1, window_size, batch_size)
    test = preprocessing.to_federated(test_files, 1, window_size, batch_size)

    yield train, val, test


def __training_metrics_fn() -> List[tf.keras.metrics.Metric]:
  return []


def __validation_metrics_fn() -> List[tf.keras.metrics.Metric]:
  thresholds = list(np.linspace(0, 1, 200, endpoint=False))
  return [
    metrics.AUC(from_logits=True, curve='PR', name='auc'),
    metrics.Precision(from_logits=True, thresholds=thresholds, name='precision'),
    metrics.Recall(from_logits=True, thresholds=thresholds, name='recall')
  ]


def __client_optimizer_fn(learning_rate: float) -> tf.keras.optimizers.Optimizer:
  return tf.keras.optimizers.Adam(learning_rate)


def __server_optimizer_fn() -> tf.keras.optimizers.Optimizer:
  return tf.keras.optimizers.SGD(1.0, momentum=.9)


def __train_step(
  server_state: ServerState,
  client_states: Dict[int, ClientState],
  dataset: FederatedDataset,
  clients_per_round: int,
  client_idx2id: Dict[int, int],
  training_fn: Callable[[ServerState, List[tf.data.Dataset], List[ClientState]], Tuple[ServerState, Metrics, List[ClientState]]]
) -> Tuple[ServerState, Metrics]:
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


def __fit(
  state: Tuple[ServerState, Dict[int, ClientState]],
  _: int,
  train_step_fn: Callable[[ServerState, Dict[int, ClientState]], Tuple[ServerState, Metrics, Dict[int, ClientState]]]
) -> Tuple[ServerState, Dict[int, ClientState]]:
  next_state, _, next_client_states = train_step_fn(state[0], state[1])
  
  return next_state, next_client_states


def __validation_step(
  weights: tff.learning.ModelWeights,
  client_states: Dict[int, ClientState],
  dataset: FederatedDataset,
  validation_fn: Callable[[tff.learning.ModelWeights, List[tf.data.Dataset], List[ClientState]], Tuple[tf.Tensor, Dict[Text, tf.Tensor]]]
) -> Tuple[tf.Tensor, Dict[Text, tf.Tensor]]:
  return validation_fn(
    weights,
    [dataset.data[client] for client in dataset.clients],
    [client_states[client] for client in dataset.clients]
  )


def __evaluate(
  confusion_matrix: tf.Tensor,
  auc: float,
  precision: np.ndarray,
  recall: np.ndarray
) -> None:
  # Confusion Matrix
  fig, ax = plt.subplots(figsize=(16, 8))

  sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap=sns.color_palette("Blues"), ax=ax)
  
  ax.set_xlabel('Predicted')
  ax.set_ylabel('Ground Truth')

  mlflow.log_figure(fig, f'confusion_matrix.png')
  plt.close(fig)

  # AUC
  mlflow.log_metric('val_auc', auc)

  # Precision Recall
  fig, ax = plt.subplots(figsize=(16, 8))

  sns.lineplot(x=recall, y=precision, ax=ax)

  ax.set_xlabel('Recall')
  ax.set_xlim(0., 1.)

  ax.set_ylabel('Precision')
  ax.set_ylim(0., 1.)

  mlflow.log_figure(fig, f'precision_recall.png')
  plt.close(fig)


def run(
  experiment_name: str,
  run_name: str,
  setup_fn: Callable[[int, int, float, float, Callable, Callable, Callable, Callable], Tuple[Callable, Callable, Callable, Callable]],
  config: Config
) -> None:
  mlflow.set_experiment(experiment_name)

  client_state_fn, iterator, _, evaluator = setup_fn(
    config.window_size,
    config.hidden_size,
    config.dropout,
    config.pos_weight,
    __training_metrics_fn,
    __validation_metrics_fn,
    partial(__client_optimizer_fn, learning_rate=config.learning_rate),
    __server_optimizer_fn
  )

  def reduce_fn(
    state: Tuple[tf.Tensor, List, List, List],
    data: Tuple[FederatedDataset, FederatedDataset, FederatedDataset]
  ):
    client_idx2id = list(data[0].clients.union(data[1].clients))
    client_states = {i: client_state_fn(idx) for idx, i in enumerate(client_idx2id)}

    train_step = partial(
      __train_step,
      dataset=data[0],
      clients_per_round=config.clients_per_round,
      client_idx2id=client_idx2id,
      training_fn=iterator.next
    )

    fitting_fn = partial(
      __fit,
      train_step_fn=train_step
    )

    validation_step = partial(
      __validation_step,
      dataset=data[1],
      validation_fn=evaluator
    )

    server_state, client_states = reduce(
      fitting_fn,
      range(1, config.rounds + 1),
      (iterator.initialize(), client_states)
    )

    confusion_matrix, metrics = validation_step(server_state.model, client_states)

    return (
      state[0] + confusion_matrix,
      state[1] + [metrics['auc']],
      state[2] + [metrics['precision']],
      state[3] + [metrics['recall']]
    )

  with mlflow.start_run(run_name=run_name):
    mlflow.log_params(config._asdict())

    confusion_matrix, auc, precision, recall = reduce(
      reduce_fn,
      __load_data(config.path, config.epochs, config.window_size, config.batch_size),
      (
        tf.zeros((2, 2), dtype=tf.int32),
        [],
        [],
        []
      )
    )

    __evaluate(
      confusion_matrix,
      np.mean(auc),
      np.mean(precision, axis=0),
      np.mean(recall, axis=0)
    )
