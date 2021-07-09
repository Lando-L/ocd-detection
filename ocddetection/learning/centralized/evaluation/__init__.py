from collections import namedtuple
from functools import partial, reduce
import os
from typing import Iterable, List, Tuple

import matplotlib.pylab as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from ocddetection import losses, metrics, models
from ocddetection.data import preprocessing, SENSORS


Config = namedtuple(
  'Config',
  ['path', 'learning_rate', 'epochs', 'batch_size', 'window_size', 'pos_weight', 'hidden_size', 'dropout']
)


def __load_data(path, window_size, batch_size) -> Iterable[Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]]:
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

    train = preprocessing.to_centralised(train_files, window_size, batch_size)
    val = preprocessing.to_centralised(val_files, window_size, batch_size)
    test = preprocessing.to_centralised(test_files, window_size, batch_size)

    yield train, val, test


def __model_fn(window_size: int, hidden_size: int, dropout: float) -> tf.keras.Model:     
  return models.bidirectional(window_size, len(SENSORS), hidden_size, dropout)


def __metrics_fn() -> List[tf.keras.metrics.Metric]:
  thresholds = list(np.linspace(0, 1, 200, endpoint=False))

  return [
    metrics.AUC(from_logits=True, curve='PR', name='auc'),
    metrics.Precision(from_logits=True, thresholds=thresholds, name='precision'),
    metrics.Recall(from_logits=True, thresholds=thresholds, name='recall')
  ]


def __optimizer_fn(learning_rate: float) -> tf.keras.optimizers.Optimizer:
  return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)


def __train_step(
  X: tf.Tensor,
  y: tf.Tensor,
  model: tf.keras.Model,
  optimizer: tf.keras.optimizers.Optimizer,
  loss_fn: tf.keras.losses.Loss
) -> None:
  with tf.GradientTape() as tape:
    logits = model(X, training=True)
    loss_value = loss_fn(y, logits)

  optimizer.apply_gradients(
    zip(
      tape.gradient(loss_value, model.trainable_variables),
      model.trainable_variables
    )
  )


def __validation_step(
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

  with mlflow.start_run(run_name=run_name):
    mlflow.log_params(config._asdict())

    def reduce_fn(state, data):
      model = __model_fn(config.window_size, config.hidden_size, config.dropout)
      loss_fn = losses.WeightedBinaryCrossEntropy(config.pos_weight)
      optimizer = __optimizer_fn(config.learning_rate)

      input_spec = (
        tf.TensorSpec((None, config.window_size, len(SENSORS)), dtype=tf.float32),
        tf.TensorSpec((None, 1), dtype=tf.float32)
      )

      train_step = tf.function(
        partial(__train_step, model=model, optimizer=optimizer, loss_fn=loss_fn),
        input_signature=input_spec
      )

      val_state = tf.zeros((2, 2), dtype=tf.int32)
      val_metrics = __metrics_fn()
      val_step = partial(__validation_step, model=model, metrics=val_metrics)

      for _ in range(1, config.epochs + 1):
        for X, y in data[0]:
          train_step(X, y)
      
      confusion_matrix = data[1].reduce(val_state, val_step)
      auc = val_metrics[0].result().numpy()
      precision = val_metrics[1].result().numpy()
      recall = val_metrics[2].result().numpy()

      return (
        state[0] + confusion_matrix,
        state[1] + [auc],
        state[2] + [precision],
        state[3] + [recall]
      )

    confusion_matrix, auc, precision, recall = reduce(
      reduce_fn,
      __load_data(config.path, config.window_size, config.batch_size),
      (
        tf.zeros((2, 2), dtype=tf.int32),
        [],
        [],
        []
      )
    )

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(16, 8))

    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap=sns.color_palette("Blues"), ax=ax)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')

    mlflow.log_figure(fig, 'confusion_matrix.png')
    plt.close(fig)

    # AUC
    mlflow.log_metric('val_auc', np.mean(auc))

    # Precision Recall
    fig, ax = plt.subplots(figsize=(16, 8))

    sns.lineplot(x=np.mean(recall, axis=0), y=np.mean(precision, axis=0), ax=ax)

    ax.set_xlabel('Recall')
    ax.set_xlim(0., 1.)

    ax.set_ylabel('Precision')
    ax.set_ylim(0., 1.)

    mlflow.log_figure(fig, 'precision_recall.png')
    plt.close(fig)
