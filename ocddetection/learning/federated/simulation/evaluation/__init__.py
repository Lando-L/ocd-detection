from collections import namedtuple
from functools import partial
import os
from typing import Callable, List, Text, Tuple

import matplotlib.pylab as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_federated as tff

from ocddetection import metrics
from ocddetection.data import preprocessing
from ocddetection.types import FederatedDataset
from ocddetection.learning.federated.simulation import common


Config = namedtuple(
  'Config',
  ['path', 'output', 'batch_size', 'window_size', 'hidden_size']
)

def __load_data(path: Text, window_size: int, batch_size: int) -> FederatedDataset:
  files = pd.Series(
    [os.path.join(path, f'S{subject}-ADL4-AUGMENTED.csv') for subject in range(1, 5)],
    index=list(range(1, 5)),
    name='path'
  )

  return preprocessing.to_federated(files, 1, window_size, batch_size)


def __optimizer_fn() -> tf.keras.optimizers.Optimizer:
  return tf.keras.optimizers.SGD(1.0, momentum=.9)


def __metrics_fn() -> List[tf.keras.metrics.Metric]:
  thresholds = list(np.linspace(0, 1, 200, endpoint=False))

  return [
    metrics.AUC(from_logits=True, curve='PR', name='auc'),
    metrics.Precision(from_logits=True, thresholds=thresholds, name='precision'),
    metrics.Recall(from_logits=True, thresholds=thresholds, name='recall')
  ]


def __evaluation_step(
  state: tf.Tensor,
  batch: Tuple[tf.Tensor, tf.Tensor],
  model: tff.learning.Model
) -> tf.Tensor:
  outputs = model.forward_pass(batch, training=False)
  y_true = tf.reshape(batch[1], (-1,))
  y_pred = tf.round(tf.nn.sigmoid(tf.reshape(outputs.predictions, (-1,))))

  return state + tf.math.confusion_matrix(y_true, y_pred, num_classes=2)


def run(
  experiment_name: str,
  run_name: str,
  setup_fn: Callable[[int, int, Callable, Callable], Tuple[List, Callable, Callable]],
  config: Config
) -> None:
  mlflow.set_experiment(experiment_name)
  
  val = __load_data(config.path, config.window_size, config.batch_size)

  server_state, client_state_fn, model_fn = setup_fn(
    config.window_size,
		config.hidden_size,
    __optimizer_fn,
    __metrics_fn
  )

  ckpt_manager = tff.simulation.FileCheckpointManager(config.output)
  ckpt = ckpt_manager.load_latest_checkpoint([server_state, [client_state_fn(-1) for _ in val.clients]])[0]

  weights = ckpt[0].model
  client_states = dict(zip(val.clients, ckpt[1]))

  with mlflow.start_run(run_name=run_name):
    mlflow.log_params(config._asdict())

    for client in val.clients:
      model = model_fn(weights, client_states[client])
      metrics = __metrics_fn()

      # Evaluation
      confusion_matrix = val.data[client].reduce(
        tf.zeros((2, 2), dtype=tf.int32),
        partial(__evaluation_step, model=model)
      )

      report = model.report_local_outputs()
      tf.nest.map_structure(lambda v, t: v.assign(t), metrics[0], report['auc'])
      tf.nest.map_structure(lambda v, t: v.assign(t), metrics[1], report['precision'])
      tf.nest.map_structure(lambda v, t: v.assign(t), metrics[2], report['recall'])

      # Confusion matrix
      fig, ax = plt.subplots(figsize=(16, 8))

      sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap=sns.color_palette("Blues"), ax=ax)
      
      ax.set_xlabel('Predicted')
      ax.set_ylabel('Ground Truth')

      mlflow.log_figure(fig, f'confusion_matrix_{client}.png')
      plt.close(fig)

      # AUC
      mlflow.log_metric(f'auc_{client}', metrics[0].result().numpy())

      # Precision Recall
      fig, ax = plt.subplots(figsize=(16, 8))
      sns.lineplot(x=metrics[1].result().numpy(), y=metrics[2].result().numpy(), ax=ax)

      ax.set_xlabel('Recall')
      ax.set_xlim(0., 1.)

      ax.set_ylabel('Precision')
      ax.set_ylim(0., 1.)

      mlflow.log_figure(fig, f'precision_recall_{client}.png')
      plt.close(fig)
