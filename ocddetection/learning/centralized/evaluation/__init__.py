from collections import namedtuple
from functools import partial
import os
from typing import List, Tuple

import matplotlib.pylab as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from ocddetection import metrics, models
from ocddetection.data import preprocessing, SENSORS
from ocddetection.types import FederatedDataset


Config = namedtuple(
  'Config',
  ['path', 'output', 'batch_size', 'window_size', 'hidden_size']
)


def __load_data(path, window_size, batch_size) -> FederatedDataset:
  files = pd.Series(
    [os.path.join(path, f'S{subject}-ADL4-AUGMENTED.csv') for subject in range(1, 5)],
    index=list(range(1, 5)),
    name='path'
  )

  return preprocessing.to_federated(files, 1, window_size, batch_size)


def __model_fn(window_size: int, hidden_size: int) -> tf.keras.Model:     
  return models.bidirectional(window_size, len(SENSORS), hidden_size, dropout=0.0, pos_weight=1.0)


def __metrics_fn() -> List[tf.keras.metrics.Metric]:
  thresholds = list(np.linspace(0, 1, 200))

  return [
    metrics.AUC(from_logits=True, curve='PR', name='auc'),
    metrics.Precision(from_logits=True, thresholds=thresholds, name='precision'),
    metrics.Recall(from_logits=True, thresholds=thresholds, name='recall')
  ]


def __evaluation_step(
  state: tf.Tensor,
  batch: Tuple[tf.Tensor, tf.Tensor],
  model: tf.keras.Model,
  metrics: List[tf.keras.metrics.Metric]
) -> tf.Tensor:
  logits = model(batch[0], training=False)
  y_true = tf.reshape(batch[1], (-1,))
  y_pred = tf.round(tf.nn.sigmoid(tf.reshape(logits, (-1,))))
  
  for metric in metrics:
    metric.update_state(batch[1], logits)
  
  return state + tf.math.confusion_matrix(y_true, y_pred, num_classes=2)


def run(experiment_name: str, run_name: str, config: Config) -> None:
  mlflow.set_experiment(experiment_name)
  
  val = __load_data(config.path, config.window_size, config.batch_size)
  model = __model_fn(config.window_size, config.hidden_size)

  ckpt = tf.train.Checkpoint(model=model)
  ckpt_manager = tf.train.CheckpointManager(ckpt, config.output, max_to_keep=5)

  if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

  metrics = __metrics_fn()
  step = partial(__evaluation_step, model=model, metrics=metrics)
  state = tf.zeros((2, 2), dtype=tf.int32)

  with mlflow.start_run(run_name=run_name):
    mlflow.log_params(config._asdict())

    for client in val.clients:
      for metric in metrics:
        metric.reset_states()

      # Evaluation
      confusion_matrix = val.data[client].reduce(state, step)

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
